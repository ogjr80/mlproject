# Eclat

# What this script does: uses the same 7,501 shopping baskets as Apriori, but only
# looks for sets of items that are often bought together (no "if A then B" direction).
# Eclat only cares about support: how often the items appear together.
# Data: Market_Basket_Optimisation.csv (one basket per row, up to 20 items each).
# Python version: ../Python/eclat.ipynb

# Data Preprocessing
# arules is the R package for association rules. install.packages() only needs to run once.
# The read.csv line just lets you peek at the raw file.
# read.transactions turns the file into the special "basket list" format arules needs.
# sep = ',' says items are separated by commas. rm.duplicates = TRUE removes an item
# listed twice in the same basket.
# summary() shows basket and item counts; itemFrequencyPlot(topN = 10) charts the top 10 items.
# install.packages('arules')
library(arules)
dataset = read.csv('Market_Basket_Optimisation.csv')
dataset = read.transactions('Market_Basket_Optimisation.csv', sep = ',', rm.duplicates = TRUE)
summary(dataset)
itemFrequencyPlot(dataset, topN = 10)

# Training Eclat on the dataset
#   support = 0.003: the items appear together in at least 0.3% of baskets (about 23 of 7,501).
#   minlen = 2: only keep sets of at least 2 items. A single item on its own isn't a pattern.
# There is no confidence setting, because Eclat doesn't make "if A then B" rules.
rules = eclat(data = dataset, parameter = list(support = 0.003, minlen = 2))

# Visualising the results
# sort(rules, by = 'support') puts the most common sets first, [1:10] keeps the top 10,
# and inspect() prints them as a readable table.
inspect(sort(rules, by = 'support')[1:10])