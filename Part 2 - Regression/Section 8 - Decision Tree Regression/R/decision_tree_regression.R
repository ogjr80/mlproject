# Decision Tree Regression

# What this script does: it predicts the salary for a job level (such as 6.5)
# with a decision tree. A tree works like a game of 20 Questions: it asks
# yes/no questions such as "Is the level below 6.5?" and each final group (a
# leaf) predicts the average salary of the rows in it. The result is a staircase.
# Data: 'Position_Salaries.csv' with 10 jobs and columns Position, Level (1-10)
# and Salary (from $45,000 at level 1 to $1,000,000 at level 10).
# Python version: ../Python/decision_tree_regression.ipynb

# Importing the dataset
# dataset[2:3] keeps only columns 2 and 3 (Level and Salary). Position is
# dropped because the job title says the same thing as the level number.
dataset = read.csv('Position_Salaries.csv')
dataset = dataset[2:3]

# Splitting the dataset into the Training set and Test set
# Left commented out on purpose: with only 10 rows there is too little data to
# hold any back, so the model learns from all of it.
# # install.packages('caTools')
# library(caTools)
# set.seed(123)
# split = sample.split(dataset$Salary, SplitRatio = 2/3)
# training_set = subset(dataset, split == TRUE)
# test_set = subset(dataset, split == FALSE)

# Feature Scaling
# Not needed: a tree only asks "is this value above or below a cut-off?",
# so the size of the numbers doesn't matter.
# training_set = scale(training_set)
# test_set = scale(test_set)

# Fitting Decision Tree Regression to the dataset
# rpart ("recursive partitioning") is R's decision tree package.
# Salary ~ . means "explain Salary using every other column" (just Level).
# By default rpart won't split a group with fewer than 20 rows, and we only
# have 10, so it would make no splits at all. minsplit = 1 lets it keep
# splitting down to single rows.
# install.packages('rpart')
library(rpart)
regressor = rpart(formula = Salary ~ .,
                  data = dataset,
                  control = rpart.control(minsplit = 1))

# Predicting a new result with Decision Tree Regression
# predict() needs a data frame with the same column name used in training (Level).
y_pred = predict(regressor, data.frame(Level = 6.5))

# Visualising the Decision Tree Regression results (higher resolution)
# A tree's prediction is flat inside each leaf and jumps at the cut-offs. To
# see those steps clearly we predict on a fine grid of levels 1.00, 1.01 ...
# 10.00 (step 0.01), rather than joining only the 10 real levels.
# ggplot layers: red dots = real salaries, blue line = the tree's predictions.
# install.packages('ggplot2')
library(ggplot2)
x_grid = seq(min(dataset$Level), max(dataset$Level), 0.01)
ggplot() +
  geom_point(aes(x = dataset$Level, y = dataset$Salary),
             colour = 'red') +
  geom_line(aes(x = x_grid, y = predict(regressor, newdata = data.frame(Level = x_grid))),
            colour = 'blue') +
  ggtitle('Truth or Bluff (Decision Tree Regression)') +
  xlab('Level') +
  ylab('Salary')

# Plotting the tree
# plot() draws the branches and text() writes the questions and leaf values on them.
plot(regressor)
text(regressor)