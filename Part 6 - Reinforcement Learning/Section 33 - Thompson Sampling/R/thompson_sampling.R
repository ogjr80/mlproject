# Thompson Sampling

# What this script does: the same 10-ads problem as UCB. Each of 10,000 visitors sees
# one ad, and we want to find the best ad quickly. Thompson Sampling keeps a
# "belief" about each ad's click rate, draws a random guess from each belief,
# and shows the ad with the highest guess. Beliefs sharpen as clicks come in.
# Data: Ads_CTR_Optimisation.csv (10,000 rows x 10 ads; 1 = this visitor would click this ad).
# Python version: ../Python/thompson_sampling.ipynb

# Importing the dataset
# The table lets us simulate real visitors. Each round the algorithm only
# "sees" the one cell for the ad it chose.
dataset = read.csv('Ads_CTR_Optimisation.csv')

# Implementing Thompson Sampling
# N = 10,000 rounds (visitors), d = 10 ads.
# integer(d) makes a list of d zeros, one slot per ad:
#   numbers_of_rewards_1[i] = times ad i was shown and clicked (wins),
#   numbers_of_rewards_0[i] = times ad i was shown and not clicked (losses).
N = 10000
d = 10
ads_selected = integer(0)
numbers_of_rewards_1 = integer(d)
numbers_of_rewards_0 = integer(d)
total_reward = 0
for (n in 1:N) {
  ad = 0
  max_random = 0
  for (i in 1:d) {
    # rbeta(n = 1, ...) draws one random number between 0 and 1 from a "beta"
    # distribution: a guess at ad i's click rate. shape1 = wins + 1, shape2 = losses + 1.
    # Few showings = wide, uncertain guesses (so the ad still gets tried);
    # many showings = guesses close to the true rate.
    random_beta = rbeta(n = 1,
                        shape1 = numbers_of_rewards_1[i] + 1,
                        shape2 = numbers_of_rewards_0[i] + 1)
    if (random_beta > max_random) {
      max_random = random_beta
      ad = i
    }
  }
  # Show the ad with the highest guess, then add a win or a loss to its count.
  ads_selected = append(ads_selected, ad)
  reward = dataset[n, ad]
  if (reward == 1) {
    numbers_of_rewards_1[ad] = numbers_of_rewards_1[ad] + 1
  } else {
    numbers_of_rewards_0[ad] = numbers_of_rewards_0[ad] + 1
  }
  total_reward = total_reward + reward
}

# Visualising the results
# hist() draws a bar per ad showing how many times it was shown.
# One very tall bar (ad 5) means Thompson Sampling found a clear winner.
# Its total_reward is usually higher than UCB's and far above random selection.
hist(ads_selected,
     col = 'blue',
     main = 'Histogram of ads selections',
     xlab = 'Ads',
     ylab = 'Number of times each ad was selected')