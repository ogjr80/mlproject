"""Generate a synthetic mobile-payments network with planted fraud patterns.

This is a *synthetic* payments network modelled on a mobile-payments company
(customers, merchants and cash-in / cash-out agents). No real customer data is
used. Everything is drawn from random distributions with a fixed seed, so the
same command always produces the same files.

Usage (from this folder):
    python generate_payments_graph.py                  # default size: 200,000 accounts, ~2M transactions
    python generate_payments_graph.py --accounts 1000000   # a much bigger network

Outputs (next to this script unless --out is given):
    accounts.csv.gz      one row per account (the nodes of the graph)
    transactions.csv.gz  one row per payment (the edges of the graph)

Planted fraud patterns (labelled in accounts.role / accounts.is_fraud and
transactions.is_suspicious):
    (a) fraud rings      - small groups of mostly new accounts that share phones
                           and pass money around among themselves
    (b) money mules      - many scam victims -> mule -> collector / quick cash-out
    (c) laundering loops - A -> B -> C -> A with almost identical amounts
    (d) account takeover - an old, quiet account suddenly sends night-time
                           transfers to brand-new "drop" accounts, which cash out
Each fraud account also has ordinary everyday activity, so its own numbers look
fairly normal: the giveaway is mostly *who it transacts with*.
"""
import argparse
import os

import numpy as np
import pandas as pd

DAYS = 90
START = pd.Timestamp('2025-01-01')
REGIONS = np.array(['Capital', 'Coast', 'Lakeside', 'Highlands', 'Rift Valley', 'Western', 'North', 'Abroad'])
REGION_P = np.array([0.30, 0.13, 0.13, 0.14, 0.12, 0.10, 0.06, 0.02])
CHANNELS = np.array(['P2P', 'merchant_payment', 'cash_in', 'cash_out', 'bill_pay', 'international'])
P2P, MERCHANT, CASH_IN, CASH_OUT, BILL, INTL = range(6)

# Share of transactions per hour of day for ordinary customers (sums to 1 after normalising).
HOUR_W = np.array([0.4, 0.25, 0.2, 0.2, 0.3, 0.8, 2.0, 3.8, 5.2, 5.8, 6.0, 6.2,
                   6.6, 6.4, 6.0, 5.9, 6.0, 6.4, 6.8, 6.6, 5.4, 4.0, 2.4, 1.2])
HOUR_P = HOUR_W / HOUR_W.sum()
NIGHT_W = HOUR_W.copy(); NIGHT_W[[22, 23, 0, 1, 2, 3, 4]] *= 6.0
NIGHT_P = NIGHT_W / NIGHT_W.sum()


def lognormal(rng, median, sigma, size):
    return np.round(np.exp(np.log(median) + sigma * rng.standard_normal(size)), 2)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--accounts', type=int, default=200_000, help='number of accounts (nodes)')
    ap.add_argument('--tx-per-account', type=float, default=9.5, help='average everyday transactions per account')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--out', default=os.path.dirname(os.path.abspath(__file__)))
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    N = args.accounts

    # ------------------------------------------------------------------ accounts
    acc_type = rng.choice(np.array(['customer', 'merchant', 'agent']), size=N, p=[0.935, 0.05, 0.015])
    is_cust, is_merch, is_agent = acc_type == 'customer', acc_type == 'merchant', acc_type == 'agent'
    cust_ids = np.flatnonzero(is_cust)

    # Roles: everyone starts normal; fraud roles are assigned to customers below.
    role = np.full(N, 'normal', dtype=object)
    pool = rng.permutation(cust_ids)  # shuffled customers to draw fraudsters from
    ptr = 0

    def take(k):
        nonlocal ptr
        out = pool[ptr:ptr + k]; ptr += k
        return out

    # (a) fraud rings
    n_rings = max(2, N // 1500)
    ring_sizes = rng.integers(5, 13, n_rings)
    rings = [take(s) for s in ring_sizes]
    for r in rings: role[r] = 'fraud_ring'
    # (b) mule networks: one collector + several mules each
    n_net = max(2, N // 1500)
    collectors = take(n_net); role[collectors] = 'mule_collector'
    net_mules = [take(int(rng.integers(4, 9))) for _ in range(n_net)]
    for m in net_mules: role[m] = 'mule'
    # (c) laundering loops
    n_cyc = max(2, N // 2000)
    cycles = [take(int(rng.integers(3, 7))) for _ in range(n_cyc)]
    for c in cycles: role[c] = 'launderer'
    # (d) account takeover: compromised accounts + new "drop" accounts that receive the stolen money
    n_ato = max(2, N // 600)
    compromised = take(n_ato); role[compromised] = 'account_takeover'
    drops = take(max(2, N // 1000)); role[drops] = 'mule'
    is_fraud = (role != 'normal').astype(np.int8)

    # Account age: most accounts are years old, some are brand new. Fraud roles lean newer,
    # but plenty of normal accounts are new too, so age alone is a weak clue.
    age = np.where(rng.random(N) < 0.12, rng.integers(1, 90, N),
                   np.clip(np.exp(np.log(600) + 0.85 * rng.standard_normal(N)), 30, 3650)).astype(int)
    age[~is_cust] = np.clip(age[~is_cust] * 1.6, 60, 3650).astype(int)
    def make_new(ids, p):
        sel = ids[rng.random(len(ids)) < p]
        age[sel] = rng.integers(1, 90, len(sel))
    for r in rings: make_new(r, 0.35)
    make_new(np.concatenate(net_mules), 0.18); make_new(collectors, 0.25); make_new(drops, 0.55)
    age[compromised] = np.maximum(age[compromised], rng.integers(400, 2500, n_ato))

    new = age < 90
    kyc = np.where(new, rng.choice([1, 2, 3], N, p=[0.55, 0.33, 0.12]), rng.choice([1, 2, 3], N, p=[0.15, 0.50, 0.35]))
    kyc[~is_cust] = np.maximum(kyc[~is_cust], 2)
    region = rng.choice(REGIONS, N, p=REGION_P)
    region[~is_cust & (region == 'Abroad')] = 'Capital'
    device_count = 1 + rng.poisson(0.35, N)
    is_business = np.where(is_cust, rng.random(N) < 0.03, rng.random(N) < 0.92).astype(np.int8)
    for c in cycles: is_business[c] = (rng.random(len(c)) < 0.25).astype(np.int8)
    balance = np.where(is_business == 1, lognormal(rng, 25000, 1.1, N), lognormal(rng, 1800, 1.15, N))
    signup = rng.choice(np.array(['app', 'USSD', 'agent', 'web']), N, p=[0.55, 0.25, 0.15, 0.05])

    # Phones (devices): 10% of customers share a phone with family members (households of 2-4).
    device = np.arange(N, dtype=np.int64) + 1_000_000
    fam = rng.permutation(cust_ids[rng.random(len(cust_ids)) < 0.10])
    i = 0
    while i < len(fam) - 1:
        k = int(rng.integers(2, 5)); device[fam[i:i + k]] = device[fam[i]]; i += k
    # Fraud rings run several accounts from a handful of phones.
    for r in rings:
        phones = device[rng.choice(r, int(rng.integers(1, 4)), replace=False)]
        sel = r[rng.random(len(r)) < 0.7]
        device[sel] = rng.choice(phones, len(sel))
    # Drop accounts are often opened in pairs on one phone.
    d = rng.permutation(drops)
    device[d[1::2][:len(d) // 2]] = device[d[0::2][:len(d) // 2]]

    # ------------------------------------------------- everyday (normal) activity
    # Friendship graph by preferential attachment: each new customer links to a few
    # existing customers, preferring ones that already have many links -> power-law degrees.
    order = cust_ids[np.argsort(-age[cust_ids], kind='stable')]
    deg = np.zeros(N, dtype=np.int64)
    m_links = np.clip(rng.geometric(0.33, N), 1, 12)
    src_l, dst_l = [], []
    B = 2000
    first = order[:B]
    a = rng.choice(first, B * 2); b = rng.choice(first, B * 2)
    src_l.append(a); dst_l.append(b); np.add.at(deg, a, 1); np.add.at(deg, b, 1)
    for s in range(B, len(order), B):
        batch, earlier = order[s:s + B], order[:s]
        reps = np.repeat(batch, m_links[batch])
        w = (deg[earlier] + 1).astype(float)
        tgt = earlier[rng.choice(len(earlier), len(reps), p=w / w.sum())]
        src_l.append(reps); dst_l.append(tgt)
        np.add.at(deg, reps, 1); np.add.at(deg, tgt, 1)
    fs, fd = np.concatenate(src_l), np.concatenate(dst_l)
    keep = fs != fd
    fs, fd = np.concatenate([fs[keep], fd[keep]]), np.concatenate([fd[keep], fs[keep]])
    o = np.argsort(fs, kind='stable'); fs, fd = fs[o], fd[o]
    f_start = np.searchsorted(fs, np.arange(N)); f_deg = np.bincount(fs, minlength=N)

    merchants, agents = np.flatnonzero(is_merch), np.flatnonzero(is_agent)
    m_pop = rng.pareto(1.3, len(merchants)) + 1; m_pop /= m_pop.sum()
    a_pop = rng.pareto(1.5, len(agents)) + 1; a_pop /= a_pop.sum()
    billers = merchants[np.argsort(-m_pop)[:12]]
    abroad = np.flatnonzero(region == 'Abroad')
    fav_merch = merchants[rng.choice(len(merchants), (N, 3), p=m_pop)]
    home_agent = agents[rng.choice(len(agents), N, p=a_pop)]
    night_owl = rng.random(N) < 0.08   # night-shift workers, bar staff, ...

    activity = np.exp(1.1 * rng.standard_normal(N))
    activity[~is_cust] *= 0.6
    M = int(args.tx_per_account * N)
    m_cust, m_cashin = int(M * 0.85), int(M * 0.12)
    m_merch = M - m_cust - m_cashin

    snd, rcv, chan = [], [], []
    # customer-initiated
    w = np.where(is_cust, activity, 0); s = rng.choice(N, m_cust, p=w / w.sum())
    ch = rng.choice(np.array([P2P, MERCHANT, CASH_OUT, BILL, INTL]), m_cust, p=[0.40, 0.30, 0.15, 0.11, 0.04])
    ch[(ch == P2P) & (f_deg[s] == 0)] = MERCHANT
    r = np.empty(m_cust, dtype=np.int64)
    k = ch == P2P
    idx = f_start[s[k]] + np.floor(f_deg[s[k]] * rng.random(k.sum()) ** 2).astype(int)
    r[k] = fd[idx]
    k = ch == MERCHANT
    use_fav = rng.random(k.sum()) < 0.6
    r[k] = np.where(use_fav, fav_merch[s[k], rng.integers(0, 3, k.sum())], merchants[rng.choice(len(merchants), k.sum(), p=m_pop)])
    k = ch == CASH_OUT
    r[k] = np.where(rng.random(k.sum()) < 0.8, home_agent[s[k]], agents[rng.choice(len(agents), k.sum(), p=a_pop)])
    k = ch == BILL; r[k] = rng.choice(billers, k.sum())
    k = ch == INTL; r[k] = rng.choice(abroad, k.sum())
    snd.append(s); rcv.append(r); chan.append(ch)
    # agents load e-money for customers (cash-in)
    w = np.where(is_cust, activity, 0); r = rng.choice(N, m_cashin, p=w / w.sum())
    s = np.where(rng.random(m_cashin) < 0.8, home_agent[r], agents[rng.choice(len(agents), m_cashin, p=a_pop)])
    snd.append(s); rcv.append(r); chan.append(np.full(m_cashin, CASH_IN))
    # merchants pay suppliers (other merchants) or cash out at agents
    s = merchants[rng.choice(len(merchants), m_merch, p=m_pop)]
    sup = rng.random(m_merch) < 0.5
    r = np.where(sup, merchants[rng.choice(len(merchants), m_merch, p=m_pop)], home_agent[s])
    c = np.where(sup, P2P, CASH_OUT)
    k = r == s; r[k] = home_agent[s[k]]; c[k] = CASH_OUT
    snd.append(s); rcv.append(r); chan.append(c)

    snd, rcv, chan = np.concatenate(snd), np.concatenate(rcv), np.concatenate(chan)
    n0 = len(snd)
    amt_med = np.array([1500, 850, 2500, 3000, 2200, 15000.]); amt_sig = np.array([1.0, 0.9, 0.9, 0.9, 0.6, 0.8])
    amount = np.round(np.exp(np.log(amt_med[chan]) + amt_sig[chan] * rng.standard_normal(n0)), 2)
    amount[np.isin(snd, merchants)] *= 4
    owl = night_owl[snd]
    hour = np.where(owl, rng.choice(24, n0, p=NIGHT_P), rng.choice(24, n0, p=HOUR_P))
    t = rng.integers(0, DAYS, n0) * 86400 + hour * 3600 + rng.integers(0, 3600, n0)
    susp = np.zeros(n0, dtype=np.int8)

    # ------------------------------------------------------------ fraud patterns
    F = {'s': [], 'r': [], 'c': [], 'a': [], 't': [], 'x': []}
    def add(s_, r_, c_, a_, t_, sus=1):
        sizes = [np.size(x) for x in (s_, r_, c_, a_, t_)]
        if min(sizes) == 0:
            return
        n = max(sizes)
        F['s'].append(np.broadcast_to(s_, n).astype(np.int64)); F['r'].append(np.broadcast_to(r_, n).astype(np.int64))
        F['c'].append(np.broadcast_to(c_, n).astype(np.int64)); F['a'].append(np.round(np.broadcast_to(a_, n).astype(float), 2))
        F['t'].append(np.broadcast_to(t_, n).astype(np.int64)); F['x'].append(np.full(n, sus, dtype=np.int8))
    normal_cust = cust_ids[role[cust_ids] == 'normal']
    def day_time(n, lo=0, hi=DAYS, p=HOUR_P):
        return rng.integers(lo, hi, n) * 86400 + rng.choice(24, n, p=p) * 3600 + rng.integers(0, 3600, n)

    MIXED_P = 0.5 * HOUR_P + 0.5 * NIGHT_P

    # ------------------------------------------------ honest look-alikes (normal)
    # These ordinary customers behave a lot like fraudsters when you look at them alone,
    # which is what makes fraud hard to spot from an account's own numbers.
    free = rng.permutation(normal_cust)
    # Savings groups ("chamas"): members pay the same fixed amount to one member each month, in rotation.
    n_groups = N // 400
    for g in np.split(free[:n_groups * 8], n_groups):
        g = g[:int(rng.integers(5, 9))]; L = len(g)
        contrib = float(rng.choice([500, 1000, 2000, 3000, 5000])); off = int(rng.integers(0, L))
        for m in range(3):
            rec = g[(m + off) % L]; payers = g[g != rec]; day = m * 30 + int(rng.integers(0, 6))
            add(payers, rec, P2P, contrib, day * 86400 + rng.choice(24, L - 1, p=HOUR_P) * 3600 + rng.integers(0, 3600, L - 1), sus=0)
            if rng.random() < 0.5:
                add(rec, home_agent[rec], CASH_OUT, contrib * (L - 1) * 0.9, (day + 1) * 86400 + 11 * 3600, sus=0)
    # Small traders: many regulars pay them by phone and they cash out at their agent the same day.
    for tr_ in free[n_groups * 8:n_groups * 8 + N // 150]:
        nb = int(rng.integers(6, 25)); payers = rng.choice(normal_cust, nb)
        a_ = lognormal(rng, 900, 0.8, nb); tt = day_time(nb)
        add(payers, tr_, P2P, a_, tt, sus=0)
        o_ = np.argsort(tt)
        for chunk in np.array_split(o_, max(1, nb // 4)):
            add(tr_, home_agent[tr_], CASH_OUT, a_[chunk].sum() * 0.9, int(tt[chunk].max() + rng.exponential(4) * 3600 + 600), sus=0)

    # (a) fraud rings: pass money around inside the ring, collect a few scam payments, cash out
    for r_ in rings:
        start = int(rng.integers(0, DAYS - 30)); L = len(r_)
        n = L * int(rng.integers(2, 6))
        a_ = rng.choice(r_, n); b_ = rng.choice(r_, n)
        ok = a_ != b_
        add(a_[ok], b_[ok], P2P, lognormal(rng, 3500, 0.7, ok.sum()), day_time(ok.sum(), start, start + 30, MIXED_P))
        victims = rng.choice(normal_cust, L)
        add(victims, r_, P2P, lognormal(rng, 2500, 0.8, L), day_time(L, start, start + 30))
        ring_agents = rng.choice(agents, 2, replace=False)
        nc = L
        add(rng.choice(r_, nc), rng.choice(ring_agents, nc), CASH_OUT, lognormal(rng, 4000, 0.6, nc), day_time(nc, start, start + 30, MIXED_P))

    # (b) mule networks: victims -> mule -> (collector | cash-out within hours); collector cashes out
    for col, mules in zip(collectors, net_mules):
        start = int(rng.integers(0, DAYS - 25))
        net_agents = rng.choice(agents, 3, replace=False)
        col_in = []
        for mu in mules:
            nv = int(rng.integers(3, 9))
            victims = rng.choice(normal_cust, nv)
            a_ = lognormal(rng, 2000, 0.7, nv); tv = day_time(nv, start, start + 21)
            add(victims, mu, P2P, a_, tv)
            delay = (rng.exponential(2.5, nv) * 3600).astype(int) + 300
            fwd = np.round(a_ * rng.uniform(0.85, 0.97, nv), 2)
            to_col = rng.random(nv) < 0.6
            add(np.full(to_col.sum(), mu), col, P2P, fwd[to_col], tv[to_col] + delay[to_col])
            add(np.full((~to_col).sum(), mu), rng.choice(net_agents, (~to_col).sum()), CASH_OUT, fwd[~to_col], tv[~to_col] + delay[~to_col])
            col_in.append(np.c_[fwd[to_col], tv[to_col] + delay[to_col]])
        ci = np.concatenate(col_in)
        if len(ci):
            ci = ci[np.argsort(ci[:, 1])]
            for chunk in np.array_split(ci, max(1, len(ci) // 3)):
                add(col, rng.choice(net_agents), CASH_OUT, chunk[:, 0].sum() * 0.98,
                    int(chunk[-1, 1] + rng.exponential(3) * 3600 + 600))

    # (c) laundering loops: the same amount (minus a small "fee") travels round A -> B -> C -> A
    for cyc in cycles:
        for _ in range(int(rng.integers(2, 6))):
            amt0 = float(lognormal(rng, 6000, 0.6, 1)[0]); fee = rng.uniform(0.005, 0.02)
            t0 = int(day_time(1)[0]); L = len(cyc); rot = int(rng.integers(0, L))
            order_c = np.roll(cyc, rot)
            hops = np.arange(L)
            ts = t0 + np.cumsum((rng.uniform(1, 12, L) * 3600).astype(int))
            add(order_c, np.roll(order_c, -1), P2P, amt0 * (1 - fee) ** hops, ts)

    # (d) account takeover: night-time burst from an old account to new drop accounts, then cash-out
    drop_groups = np.array_split(rng.permutation(drops), len(drops) // 2)
    for comp in compromised:
        dg = drop_groups[int(rng.integers(len(drop_groups)))]
        n = int(rng.integers(4, 10)); day = int(rng.integers(5, DAYS - 2))
        tt = day * 86400 + rng.choice([0, 1, 2, 3, 4], n) * 3600 + rng.integers(0, 3600, n)
        a_ = lognormal(rng, 3500, 0.6, n)
        rr = rng.choice(dg, n)
        add(comp, rr, P2P, a_, tt)
        add(rr, rng.choice(agents, n), CASH_OUT, np.round(a_ * 0.97, 2), tt + (rng.exponential(1.5, n) * 3600).astype(int) + 600)

    fs_ = np.concatenate(F['s']); fr_ = np.concatenate(F['r']); fc_ = np.concatenate(F['c'])
    fa_ = np.concatenate(F['a']); ft_ = np.clip(np.concatenate(F['t']), 0, DAYS * 86400 - 1)
    snd = np.concatenate([snd, fs_]); rcv = np.concatenate([rcv, fr_]); chan = np.concatenate([chan, fc_])
    amount = np.concatenate([amount, fa_]); t = np.concatenate([t, ft_])
    susp = np.concatenate([susp, np.concatenate(F['x'])])
    ok = snd != rcv
    snd, rcv, chan, amount, t, susp = snd[ok], rcv[ok], chan[ok], amount[ok], t[ok], susp[ok]
    o = np.argsort(t, kind='stable')
    snd, rcv, chan, amount, t, susp = snd[o], rcv[o], chan[o], amount[o], t[o], susp[o]

    # ------------------------------------------------------------------- save
    accounts = pd.DataFrame({
        'account_id': np.arange(N), 'account_type': acc_type, 'account_age_days': age, 'kyc_level': kyc,
        'region': region, 'device_count': device_count, 'device_id': device,
        'avg_daily_balance': np.round(balance, 2), 'is_business': is_business, 'signup_channel': signup,
        'role': role, 'is_fraud': is_fraud})
    ts = START + pd.to_timedelta(t, unit='s')
    tx = pd.DataFrame({
        'tx_id': np.arange(len(snd)), 'timestamp': ts.strftime('%Y-%m-%d %H:%M:%S'),
        'sender_id': snd, 'receiver_id': rcv, 'amount': np.round(amount, 2), 'channel': CHANNELS[chan],
        'hour': (t % 86400) // 3600, 'is_suspicious': susp})
    os.makedirs(args.out, exist_ok=True)
    pa, pt = os.path.join(args.out, 'accounts.csv.gz'), os.path.join(args.out, 'transactions.csv.gz')
    accounts.to_csv(pa, index=False, compression={'method': 'gzip', 'compresslevel': 9, 'mtime': 0})
    tx.to_csv(pt, index=False, compression={'method': 'gzip', 'compresslevel': 9, 'mtime': 0})
    print(f'{N:,} accounts ({is_fraud.sum():,} fraud = {is_fraud.mean():.2%}) -> {pa} ({os.path.getsize(pa) / 1e6:.1f} MB)')
    print(f'{len(tx):,} transactions ({susp.sum():,} suspicious = {susp.mean():.2%}) -> {pt} ({os.path.getsize(pt) / 1e6:.1f} MB)')
    print(accounts.role.value_counts().to_string())


if __name__ == '__main__':
    main()
