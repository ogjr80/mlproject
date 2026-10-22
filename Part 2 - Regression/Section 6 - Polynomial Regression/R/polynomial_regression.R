# Polynomial Regression

# What this script does: it predicts the salary for a job level, including
# in-between levels like 6.5. Salaries here creep up and then shoot up at the
# top, so a straight line fits badly. Polynomial regression adds level squared,
# cubed and to the 4th power so the line can curve. A straight-line model is
# built too, for comparison.
# Data: 'Position_Salaries.csv' with 10 jobs and columns Position, Level (1-10)
# and Salary (from $45,000 at level 1 to $1,000,000 at level 10).
# Python version: ../Python/polynomial_regression.ipynb

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
# Not needed: lm picks its own multiplier for each column.
# training_set = scale(training_set)
# test_set = scale(test_set)

# Fitting Linear Regression to the dataset
# An ordinary straight line, kept only as a baseline to compare against.
# Salary ~ . means "explain Salary using every other column" (here just Level).
lin_reg = lm(formula = Salary ~ .,
             data = dataset)

# Fitting Polynomial Regression to the dataset
# We add new columns by hand: Level squared, cubed and to the 4th power.
# For level 3 that is 9, 27 and 81. Then the SAME lm is fitted, and
# Salary ~ . now uses all four columns. Those extra powers are what let the
# line bend. (The Python version uses PolynomialFeatures(degree = 4) for this.)
dataset$Level2 = dataset$Level^2
dataset$Level3 = dataset$Level^3
dataset$Level4 = dataset$Level^4
poly_reg = lm(formula = Salary ~ .,
              data = dataset)

# Visualising the Linear Regression results
# ggplot builds a chart in layers joined with '+': red dots for the real
# salaries, a blue line for the model's predictions, then title and labels.
# The straight line misses the low and high levels badly.
# install.packages('ggplot2')
library(ggplot2)
ggplot() +
  geom_point(aes(x = dataset$Level, y = dataset$Salary),
             colour = 'red') +
  geom_line(aes(x = dataset$Level, y = predict(lin_reg, newdata = dataset)),
            colour = 'blue') +
  ggtitle('Truth or Bluff (Linear Regression)') +
  xlab('Level') +
  ylab('Salary')

# Visualising the Polynomial Regression results
# Same chart for the curved model: it follows the points far more closely.
# install.packages('ggplot2')
library(ggplot2)
ggplot() +
  geom_point(aes(x = dataset$Level, y = dataset$Salary),
             colour = 'red') +
  geom_line(aes(x = dataset$Level, y = predict(poly_reg, newdata = dataset)),
            colour = 'blue') +
  ggtitle('Truth or Bluff (Polynomial Regression)') +
  xlab('Level') +
  ylab('Salary')

# Visualising the Regression Model results (for higher resolution and smoother curve)
# The chart above only joins 10 points with straight segments. Here
# seq(min, max, 0.1) makes levels 1.0, 1.1, 1.2 ... 10.0, so the curve looks smooth.
# Because poly_reg was trained on Level2..Level4 too, data.frame() must supply
# those powers for every grid value as well.
# install.packages('ggplot2')
library(ggplot2)
x_grid = seq(min(dataset$Level), max(dataset$Level), 0.1)
ggplot() +
  geom_point(aes(x = dataset$Level, y = dataset$Salary),
             colour = 'red') +
  geom_line(aes(x = x_grid, y = predict(poly_reg,
                                        newdata = data.frame(Level = x_grid,
                                                             Level2 = x_grid^2,
                                                             Level3 = x_grid^3,
                                                             Level4 = x_grid^4))),
            colour = 'blue') +
  ggtitle('Truth or Bluff (Polynomial Regression)') +
  xlab('Level') +
  ylab('Salary')

# Predicting a new result with Linear Regression
# predict() needs a data frame with the same column names the model was
# trained on. The straight line gives a poor guess for level 6.5.
predict(lin_reg, data.frame(Level = 6.5))

# Predicting a new result with Polynomial Regression
# For the curved model we must also pass 6.5 squared, cubed and to the 4th.
# Its guess is much closer to the salaries of levels 6 and 7.
predict(poly_reg, data.frame(Level = 6.5,
                             Level2 = 6.5^2,
                             Level3 = 6.5^3,
                             Level4 = 6.5^4))