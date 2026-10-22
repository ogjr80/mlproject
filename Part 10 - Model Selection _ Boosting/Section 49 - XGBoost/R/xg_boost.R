# XGBoost

# What this script does: predicts which bank customers will leave (Exited = 1) using
# XGBoost, which builds many small decision trees one after another, each one fixing
# the mistakes of the trees before it. It then checks the model with 10-fold cross validation.
# Data: Churn_Modelling.csv (the same bank data as the ANN section).
# Python version: ../Python/xg_boost.ipynb (note: the notebook uses a breast-cancer dataset instead).

# Importing the dataset
# dataset[4:14] keeps the 10 useful inputs plus Exited (now column 11),
# dropping row number, customer ID and surname.
dataset = read.csv('Churn_Modelling.csv')
dataset = dataset[4:14]

# Encoding the categorical variables as factors
# XGBoost only works with numbers, so the word columns become codes:
# France = 1, Spain = 2, Germany = 3; Female = 1, Male = 2.
# factor() maps the words to codes and as.numeric() makes them plain numbers.
dataset$Geography = as.numeric(factor(dataset$Geography,
                                      levels = c('France', 'Spain', 'Germany'),
                                      labels = c(1, 2, 3)))
dataset$Gender = as.numeric(factor(dataset$Gender,
                                   levels = c('Female', 'Male'),
                                   labels = c(1, 2)))

# Splitting the dataset into the Training set and Test set
# set.seed(123) fixes the random split so results match every run.
# 80% of customers (8,000) for training, 20% (2,000) for testing.
# There is no feature scaling: tree models ask yes/no questions like "is age > 45?",
# so the size of the numbers doesn't matter.
# install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(dataset$Exited, SplitRatio = 0.8)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# Fitting XGBoost to the Training set
# xgboost needs a plain number matrix, not a data frame, hence as.matrix(...).
# [-11] = all inputs (every column except Exited); label = the 0/1 answers.
# nrounds = 10 builds 10 trees in a row. More rounds can learn more but may overfit.
# install.packages('xgboost')
library(xgboost)
classifier = xgboost(data = as.matrix(training_set[-11]), label = training_set$Exited, nrounds = 10)

# Predicting the Test set results
# predict() returns a score between 0 and 1 for each customer;
# (y_pred >= 0.5) turns it into TRUE (will leave) / FALSE (will stay).
y_pred = predict(classifier, newdata = as.matrix(test_set[-11]))
y_pred = (y_pred >= 0.5)

# Making the Confusion Matrix
# Rows = real answer, columns = guess. The diagonal counts the correct guesses.
cm = table(test_set[, 11], y_pred)

# Applying k-Fold Cross Validation
# createFolds(..., k = 10) cuts the training rows into 10 groups.
# lapply runs the code once per group (x = that group's rows): train XGBoost on the
# other 9 groups, test on this one, and return the accuracy.
# mean(as.numeric(cv)) averages the 10 scores for a fairer overall estimate.
# install.packages('caret')
library(caret)
folds = createFolds(training_set$Exited, k = 10)
cv = lapply(folds, function(x) {
  training_fold = training_set[-x, ]
  test_fold = training_set[x, ]
  classifier = xgboost(data = as.matrix(training_fold[-11]), label = training_fold$Exited, nrounds = 10)
  y_pred = predict(classifier, newdata = as.matrix(test_fold[-11]))
  y_pred = (y_pred >= 0.5)
  cm = table(test_fold[, 11], y_pred)
  accuracy = (cm[1,1] + cm[2,2]) / (cm[1,1] + cm[2,2] + cm[1,2] + cm[2,1])
  return(accuracy)
})
accuracy = mean(as.numeric(cv))