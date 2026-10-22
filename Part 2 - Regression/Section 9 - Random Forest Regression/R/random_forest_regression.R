# Random Forest Regression

# What this script does: it predicts the salary for a job level (such as 6.5)
# with a random forest: many decision trees, each trained on a slightly
# different random sample of the data, whose answers are averaged. Like asking
# many people for an estimate, individual mistakes tend to cancel out.
# Data: 'Position_Salaries.csv' with 10 jobs and columns Position, Level (1-10)
# and Salary (from $45,000 at level 1 to $1,000,000 at level 10).
# Python version: ../Python/random_forest_regression.ipynb

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
# Not needed: trees only ask "is this value above or below a cut-off?",
# so the size of the numbers doesn't matter.
# training_set = scale(training_set)
# test_set = scale(test_set)

# Fitting Random Forest Regression to the dataset
# set.seed(1234) fixes the random sampling so the forest is the same every run.
# Instead of a formula, randomForest is given the inputs and answer separately:
# x = dataset[-2] means "every column except column 2" (so just Level; it stays
# a table), and y = dataset$Salary is the answer column.
# ntree = 500 grows 500 trees (the Python version uses 10). More trees give
# steadier answers but take longer.
# install.packages('randomForest')
library(randomForest)
set.seed(1234)
regressor = randomForest(x = dataset[-2],
                         y = dataset$Salary,
                         ntree = 500)

# Predicting a new result with Random Forest Regression
# predict() needs a data frame with the same column name used in training (Level).
y_pred = predict(regressor, data.frame(Level = 6.5))

# Visualising the Random Forest Regression results (higher resolution)
# The forest's prediction is a staircase with many small steps (one set from
# each tree, averaged). A fine grid of levels (step 0.01) makes those steps visible.
# ggplot layers: red dots = real salaries, blue line = the forest's predictions.
# install.packages('ggplot2')
library(ggplot2)
x_grid = seq(min(dataset$Level), max(dataset$Level), 0.01)
ggplot() +
  geom_point(aes(x = dataset$Level, y = dataset$Salary),
             colour = 'red') +
  geom_line(aes(x = x_grid, y = predict(regressor, newdata = data.frame(Level = x_grid))),
            colour = 'blue') +
  ggtitle('Truth or Bluff (Random Forest Regression)') +
  xlab('Level') +
  ylab('Salary')