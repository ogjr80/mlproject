# Upper Confidence Bound

# What this script does: we have 10 adverts and want to find the one people click most,
# while wasting as few showings as possible on bad ads. Each of 10,000 visitors sees
# one ad. UCB shows the ad with the best "optimistic guess" of its click rate,
# and learns from each click as it goes.
# Data: Ads_CTR_Optimisation.csv (10,000 rows x 10 ads; 1 = this visitor would click this ad).
# Python version: ../Python/upper_confidence_bound.ipynb

# Importing the dataset
# The table lets us simulate real visitors. Each round the algorithm only
# "sees" the one cell for the ad it chose.
dataset = read.csv('Ads_CTR_Optimisation.csv')

# Implementing UCB
# N = 10,000 rounds (visitors), d = 10 ads.
# integer(d) makes a list of d zeros, one slot per ad:
#   numbers_of_selections[i] = how many times ad i has been shown so far,
#   sums_of_rewards[i] = how many clicks ad i has earned so far.
N = 10000
d = 10
ads_selected = integer(0)
numbers_of_selections = integer(d)
sums_of_rewards = integer(d)
total_reward = 0
# Each round: work out an upper bound for every ad and show the ad with the highest one.
for (n in 1:N) {
  ad = 0
  max_upper_bound = 0
  for (i in 1:d) {
    # Upper bound = average click rate + a "benefit of the doubt" bonus (delta_i).
    # The bonus is big for ads shown only a few times and shrinks as we learn more,
    # so rarely shown ads still get a fair try.
    if (numbers_of_selections[i] > 0) {
      average_reward = sums_of_rewards[i] / numbers_of_selections[i]
      delta_i = sqrt(3/2 * log(n) / numbers_of_selections[i])
      upper_bound = average_reward + delta_i
    } else {
        # An ad never shown gets a huge bound (1e400 is "infinity" in R),
        # so every ad is tried once in the first 10 rounds.
        upper_bound = 1e400
    }
    if (upper_bound > max_upper_bound) {
      max_upper_bound = upper_bound
      ad = i
    }
  }
  # Show the chosen ad, then record what happened so the next round knows more.
  ads_selected = append(ads_selected, ad)
  numbers_of_selections[ad] = numbers_of_selections[ad] + 1
  reward = dataset[n, ad]
  sums_of_rewards[ad] = sums_of_rewards[ad] + reward
  total_reward = total_reward + reward
}

# Visualising the results
# hist() draws a bar per ad showing how many times it was shown.
# One very tall bar (ad 5) means UCB settled on a winner. Compare total_reward
# with random_selection.R (about 1,200) to see the gain.
hist(ads_selected,
     col = 'blue',
     main = 'Histogram of ads selections',
     xlab = 'Ads',
     ylab = 'Number of times each ad was selected')