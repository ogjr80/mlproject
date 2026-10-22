# Simple Linear Regression

# What this script does: it predicts an employee's salary from their years of
# experience by drawing the best straight line through the data:
# Salary = (starting salary) + (raise per year x years of experience).
# It then plots the line against the training and test points.
# Data: 'Salary_Data.csv' with 30 employees and two columns, YearsExperience and Salary.
# Python version: ../Python/simple_linear_regression.ipynb

# Importing the dataset
# read.csv loads the file into a "data frame", R's name for a table.
dataset = read.csv('Salary_Data.csv')

# Splitting the dataset into the Training set and Test set
# install.packages is commented out: run it once per computer, then leave it off.
# set.seed(123) fixes the random shuffle so you get the same split every run.
# sample.split marks each row TRUE (training) or FALSE (test). SplitRatio = 2/3
# is the TRAINING share: 20 employees to learn from, 10 held back for testing.
# subset() then keeps only the rows with the chosen mark.
# install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(dataset$Salary, SplitRatio = 2/3)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# Feature Scaling
# Not needed here: linear regression adjusts its own multiplier to whatever
# scale the numbers are on, so these lines stay commented out.
# training_set = scale(training_set)
# test_set = scale(test_set)

# Fitting Simple Linear Regression to the Training set
# lm = "linear model". The formula Salary ~ YearsExperience reads as
# "explain Salary using YearsExperience". lm finds the line that is as close as
# possible to all the training points (least squares).
# Tip: run summary(regressor) to see the starting salary and raise per year.
regressor = lm(formula = Salary ~ YearsExperience,
               data = training_set)

# Predicting the Test set results
# Uses the line to guess the salary of each test employee from their experience.
# Compare y_pred with test_set$Salary to see how close the guesses are.
y_pred = predict(regressor, newdata = test_set)

# Visualising the Training set results
# ggplot builds a chart in layers joined with '+':
# geom_point = the real employees as red dots,
# geom_line = the model's predictions as a blue line,
# then a title and axis labels. aes() says which values go on the x and y axes.
library(ggplot2)
ggplot() +
  geom_point(aes(x = training_set$YearsExperience, y = training_set$Salary),
             colour = 'red') +
  geom_line(aes(x = training_set$YearsExperience, y = predict(regressor, newdata = training_set)),
            colour = 'blue') +
  ggtitle('Salary vs Experience (Training set)') +
  xlab('Years of experience') +
  ylab('Salary')

# Visualising the Test set results
# The red dots are now the 10 unseen test employees. The blue line is the SAME
# line learned from the training set (that's why it still uses training_set):
# we are checking how well it fits people it has never seen.
library(ggplot2)
ggplot() +
  geom_point(aes(x = test_set$YearsExperience, y = test_set$Salary),
             colour = 'red') +
  geom_line(aes(x = training_set$YearsExperience, y = predict(regressor, newdata = training_set)),
            colour = 'blue') +
  ggtitle('Salary vs Experience (Test set)') +
  xlab('Years of experience') +
  ylab('Salary')