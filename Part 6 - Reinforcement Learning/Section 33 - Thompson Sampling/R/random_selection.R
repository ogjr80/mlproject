# Random Selection

# What this script does: a "no brains" baseline for the ads problem. For each of
# 10,000 visitors it shows one of 10 ads picked completely at random and counts the clicks.
# Compare its total_reward with UCB and Thompson Sampling to see how much smarter they are.
# Data: Ads_CTR_Optimisation.csv (10,000 rows x 10 ads; 1 = this visitor would click this ad).
# Python version: none (this is a helper; the main notebooks are upper_confidence_bound.ipynb
# and thompson_sampling.ipynb in ../Python).

# Importing the dataset
dataset = read.csv('Ads_CTR_Optimisation.csv')

# Implementing Random Selection
# N = number of visitors (rounds), d = number of ads.
# integer(0) makes an empty list of whole numbers that we add to with append().
# sample(1:10, 1) picks one ad from 1 to 10 at random, like rolling a 10-sided die.
# dataset[n, ad] looks up whether visitor n would have clicked that ad (1 or 0).
# total_reward adds up the clicks. Random picking usually gets around 1,200.
N = 10000
d = 10
ads_selected = integer(0)
total_reward = 0
for (n in 1:N) {
  ad = sample(1:10, 1)
  ads_selected = append(ads_selected, ad)
  reward = dataset[n, ad]
  total_reward = total_reward + reward
}

# Visualising the results
# hist() draws a bar per ad showing how many times it was shown.
# With random picking the bars are all roughly the same height: nothing was learned.
hist(ads_selected,
     col = 'blue',
     main = 'Histogram of ads selections',
     xlab = 'Ads',
     ylab = 'Number of times each ad was selected')