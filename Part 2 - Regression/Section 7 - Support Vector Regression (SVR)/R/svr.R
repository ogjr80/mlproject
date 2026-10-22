# SVR

# What this script does: it predicts the salary for a job level (such as 6.5)
# with Support Vector Regression. SVR fits a smooth curve with a tolerance
# "tube" around it: points inside the tube count as close enough, and only
# points outside it push the curve around.
# Data: 'Position_Salaries.csv' with 10 jobs and columns Position, Level (1-10)
# and Salary (from $45,000 at level 1 to $1,000,000 at level 10).
# Python version: ../Python/support_vector_regression.ipynb

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
# SVR does need scaling, but unlike the Python version we don't do it by hand:
# R's svm() function scales the data (and un-scales the answers) for you.
# training_set = scale(training_set)
# test_set = scale(test_set)

# Fitting SVR to the dataset
# e1071 is the package that provides svm(). Run install.packages once if needed.
# Salary ~ . means "explain Salary using every other column" (just Level).
# type = 'eps-regression' tells svm() to predict a number (not a class);
# "eps" (epsilon) is the width of the tolerance tube.
# kernel = 'radial' (RBF) lets the curve bend smoothly.
# install.packages('e1071')
library(e1071)
regressor = svm(formula = Salary ~ .,
                data = dataset,
                type = 'eps-regression',
                kernel = 'radial')

# Predicting a new result
# predict() needs a data frame with the same column name used in training (Level).
y_pred = predict(regressor, data.frame(Level = 6.5))

# Visualising the SVR results
# ggplot builds a chart in layers joined with '+': red dots for the real
# salaries, a blue line for the model's predictions, then title and labels.
# Note how SVR treats the CEO salary (level 10) as an outlier and doesn't
# chase it: it sits outside the tube.
# install.packages('ggplot2')
library(ggplot2)
ggplot() +
  geom_point(aes(x = dataset$Level, y = dataset$Salary),
             colour = 'red') +
  geom_line(aes(x = dataset$Level, y = predict(regressor, newdata = dataset)),
            colour = 'blue') +
  ggtitle('Truth or Bluff (SVR)') +
  xlab('Level') +
  ylab('Salary')

# Visualising the SVR results (for higher resolution and smoother curve)
# seq(min, max, 0.1) makes levels 1.0, 1.1 ... 10.0, so the blue curve is
# drawn through 91 points instead of 10 and looks smooth.
# install.packages('ggplot2')
library(ggplot2)
x_grid = seq(min(dataset$Level), max(dataset$Level), 0.1)
ggplot() +
  geom_point(aes(x = dataset$Level, y = dataset$Salary),
             colour = 'red') +
  geom_line(aes(x = x_grid, y = predict(regressor, newdata = data.frame(Level = x_grid))),
            colour = 'blue') +
  ggtitle('Truth or Bluff (SVR)') +
  xlab('Level') +
  ylab('Salary')