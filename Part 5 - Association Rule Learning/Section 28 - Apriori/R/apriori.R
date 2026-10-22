# Apriori

# What this script does: looks through 7,501 shopping baskets from one week and
# finds rules like "people who buy A also tend to buy B", so a shop can place
# those items together or offer deals on them.
# Data: Market_Basket_Optimisation.csv (one basket per row, up to 20 items each).
# Python version: ../Python/apriori.ipynb

# Data Preprocessing
# arules is the R package for association rules. install.packages() only needs to run once.
# The read.csv line just lets you peek at the raw file. header = FALSE because the
# first line is already a basket, not column titles.
# read.transactions turns the file into the special "basket list" format arules needs.
# sep = ',' says items are separated by commas. rm.duplicates = TRUE removes an item
# listed twice in the same basket (R prints how many it found).
# summary() shows the number of baskets, items, and the most common items.
# itemFrequencyPlot(topN = 10) draws a bar chart of the 10 most often bought items.
# install.packages('arules')
library(arules)
dataset = read.csv('Market_Basket_Optimisation.csv', header = FALSE)
dataset = read.transactions('Market_Basket_Optimisation.csv', sep = ',', rm.duplicates = TRUE)
summary(dataset)
itemFrequencyPlot(dataset, topN = 10)

# Training Apriori on the dataset
# Each setting is a minimum bar a rule must clear:
#   support = 0.004: the items appear together in at least 0.4% of baskets
#     (about 30 of 7,501). Roughly "bought 4 times a day over 7 days" / 7,501.
#   confidence = 0.2: when a basket has the left-side item, it also has the
#     right-side item at least 20% of the time.
# The Python notebook uses support 0.003. Lower bars give more (but weaker) rules.
rules = apriori(data = dataset, parameter = list(support = 0.004, confidence = 0.2))

# Visualising the results
# There is no chart: "visualising" means printing the best rules as a table.
# sort(rules, by = 'lift') puts the strongest links first. Lift above 1 means the
# items are bought together more often than chance would suggest.
# [1:10] keeps the top 10, and inspect() prints them in readable form
# (lhs = "if they buy this", rhs = "they also buy this").
inspect(sort(rules, by = 'lift')[1:10])