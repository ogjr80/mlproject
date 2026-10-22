# Regression Template

# What this script does: it is a reusable skeleton for any regression model
# (a model that predicts a number). It loads the data, leaves a gap where you
# create your model, then predicts a new value and plots the results.
# Data: 'Position_Salaries.csv' with 10 jobs and columns Position, Level (1-10)
# and Salary.
# This is a reusable starting template.

# Importing the dataset
# dataset[2:3] keeps only columns 2 and 3 (Level and Salary). Position is
# dropped because the job title says the same thing as the level number.
dataset = read.csv('Position_Salaries.csv')
dataset = dataset[2:3]

# Splitting the dataset into the Training set and Test set
# Commented out because 10 rows is too few to hold any back. For bigger data,
# remove the '#'s: set.seed fixes the random split, sample.split marks rows
# TRUE/FALSE (SplitRatio is the TRAINING share, and splitting on the label
# column keeps a similar mix in both sets), and subset() keeps the marked rows.
# # install.packages('caTools')
# library(caTools)
# set.seed(123)
# split = sample.split(dataset$Salary, SplitRatio = 2/3)
# training_set = subset(dataset, split == TRUE)
# test_set = subset(dataset, split == FALSE)

# Feature Scaling
# Turn this on only if your chosen model needs it. scale() makes each column
# average 0 with most values between about -2 and +2.
# training_set = scale(training_set)
# test_set = scale(test_set)

# Fitting the Regression Model to the dataset
# Replace the line below with your model, for example
# regressor = lm(formula = Salary ~ ., data = dataset)
# The rest of the script expects it to be called 'regressor'.
# Create your regressor here

# Predicting a new result
# predict() needs a data frame with the same column name used in training (Level).
y_pred = predict(regressor, data.frame(Level = 6.5))

# Visualising the Regression Model results
# ggplot builds a chart in layers joined with '+': red dots for the real
# salaries, a blue line for the model's predictions, then title and labels.
# install.packages('ggplot2')
library(ggplot2)
ggplot() +
  geom_point(aes(x = dataset$Level, y = dataset$Salary),
             colour = 'red') +
  geom_line(aes(x = dataset$Level, y = predict(regressor, newdata = dataset)),
            colour = 'blue') +
  ggtitle('Truth or Bluff (Regression Model)') +
  xlab('Level') +
  ylab('Salary')

# Visualising the Regression Model results (for higher resolution and smoother curve)
# seq(min, max, 0.1) makes levels 1.0, 1.1 ... 10.0, so the blue line is drawn
# through many more points and shows the model's true shape.
# install.packages('ggplot2')
library(ggplot2)
x_grid = seq(min(dataset$Level), max(dataset$Level), 0.1)
ggplot() +
  geom_point(aes(x = dataset$Level, y = dataset$Salary),
             colour = 'red') +
  geom_line(aes(x = x_grid, y = predict(regressor, newdata = data.frame(Level = x_grid))),
            colour = 'blue') +
  ggtitle('Truth or Bluff (Regression Model)') +
  xlab('Level') +
  ylab('Salary')