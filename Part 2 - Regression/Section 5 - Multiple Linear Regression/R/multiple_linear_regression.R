# Multiple Linear Regression

# What this script does: it predicts a startup's profit from several inputs at
# once: Profit = (base) + (a x R&D) + (b x Admin) + (c x Marketing) + (an
# adjustment for the State). It learns the numbers from 40 startups and tests on 10.
# Data: '50_Startups.csv' with 50 startups and columns R.D.Spend,
# Administration, Marketing.Spend, State and Profit.
# Python version: ../Python/multiple_linear_regression.ipynb

# Importing the dataset
# read.csv loads the file into a "data frame", R's name for a table.
# R swaps spaces and '&' in column names for dots, e.g. "R&D Spend" becomes R.D.Spend.
dataset = read.csv('50_Startups.csv')

# Encoding categorical data
# factor() tells R that State is a set of categories, not numbers.
# The labels 1, 2, 3 are just names: lm will automatically turn a factor into
# yes/no "dummy" columns, so it won't think Florida is "3 times" New York.
# (This is what one-hot encoding does by hand in the Python version.)
dataset$State = factor(dataset$State,
                       levels = c('New York', 'California', 'Florida'),
                       labels = c(1, 2, 3))

# Splitting the dataset into the Training set and Test set
# install.packages is commented out: run it once per computer, then leave it off.
# set.seed(123) fixes the random shuffle so you get the same split every run.
# sample.split marks each row TRUE (training) or FALSE (test). SplitRatio = 0.8
# is the TRAINING share: 40 startups to learn from, 10 held back for testing.
# subset() then keeps only the rows with the chosen mark.
# install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(dataset$Profit, SplitRatio = 0.8)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# Feature Scaling
# Not needed: linear regression picks its own multiplier for each column,
# whatever scale it is on.
# training_set = scale(training_set)
# test_set = scale(test_set)

# Fitting Multiple Linear Regression to the Training set
# The formula Profit ~ . reads "explain Profit using every other column".
# The dot is shorthand for "all the rest", so you don't have to list them.
# lm also drops one of the State dummy columns for you, avoiding the
# "dummy variable trap" (one column being predictable from the others).
# Tip: run summary(regressor) to see each multiplier and how much it matters.
regressor = lm(formula = Profit ~ .,
               data = training_set)

# Predicting the Test set results
# Guesses the profit of each of the 10 test startups.
# Compare y_pred with test_set$Profit to see how close the guesses are.
y_pred = predict(regressor, newdata = test_set)