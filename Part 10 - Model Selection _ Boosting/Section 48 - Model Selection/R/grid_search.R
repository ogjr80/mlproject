# Grid Search

# What this script does: trains a Kernel SVM to predict who buys a product after seeing
# an ad, checks it with 10-fold cross validation, then uses grid search to try many
# settings ("hyperparameters", like the knobs on an oven) and pick the best ones.
# Data: Social_Network_Ads.csv (400 people: Age, EstimatedSalary, Purchased).
# Python version: ../Python/grid_search.ipynb

# Importing the dataset
# dataset[3:5] keeps Age, EstimatedSalary and Purchased.
dataset = read.csv('Social_Network_Ads.csv')
dataset = dataset[3:5]

# Encoding the target feature as factor
# factor() tells R that Purchased is a category (0 or 1), not a number to do maths on.
dataset$Purchased = factor(dataset$Purchased, levels = c(0, 1))

# Splitting the dataset into the Training set and Test set
# set.seed(123) fixes the random split so results match every run.
# 75% of people (300) for training, 25% (100) for testing.
# install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(dataset$Purchased, SplitRatio = 0.75)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# Feature Scaling
# [-3] = every column except column 3 (Purchased). The SVM measures distances,
# so salary (tens of thousands) would swamp age (under 100) without scaling.
training_set[-3] = scale(training_set[-3])
test_set[-3] = scale(test_set[-3])

# Fitting Kernel SVM to the Training set
# kernel = 'radial' (RBF) lets the dividing line bend and curve.
# type = 'C-classification' says we're predicting categories (buys / doesn't buy).
# install.packages('e1071')
library(e1071)
classifier = svm(formula = Purchased ~ .,
                 data = training_set,
                 type = 'C-classification',
                 kernel = 'radial')

# Predicting the Test set results
# [-3] leaves out the real answer, so the model only sees age and salary.
y_pred = predict(classifier, newdata = test_set[-3])

# Making the Confusion Matrix
# Rows = real answer, columns = guess. The diagonal counts the correct guesses.
cm = table(test_set[, 3], y_pred)

# Applying k-Fold Cross Validation
# caret's createFolds(..., k = 10) cuts the 300 training rows into 10 groups of about 30.
# lapply(folds, function(x) {...}) runs the code once per group (x = that group's rows):
# train on the other 9 groups, test on this one, and return the accuracy.
# mean(as.numeric(cv)) averages the 10 scores: a fairer estimate than a single test.
# install.packages('caret')
library(caret)
folds = createFolds(training_set$Purchased, k = 10)
cv = lapply(folds, function(x) {
  training_fold = training_set[-x, ]
  test_fold = training_set[x, ]
  classifier = svm(formula = Purchased ~ .,
                   data = training_fold,
                   type = 'C-classification',
                   kernel = 'radial')
  y_pred = predict(classifier, newdata = test_fold[-3])
  cm = table(test_fold[, 3], y_pred)
  accuracy = (cm[1,1] + cm[2,2]) / (cm[1,1] + cm[2,2] + cm[1,2] + cm[2,1])
  return(accuracy)
})
accuracy = mean(as.numeric(cv))

# Applying Grid Search to find the best parameters
# caret's train(..., method = 'svmRadial') fits a radial SVM (from the kernlab package)
# many times, each with different settings: C (how strict it is about training mistakes)
# and sigma (how curvy the boundary can be). It scores each try by resampling
# (bootstrap by default) and keeps the best.
# Printing classifier shows the score for every setting tried;
# classifier$bestTune shows the winning sigma and C.
# The charts below use this tuned model, because it replaces "classifier".
# install.packages('caret')
library(caret)
classifier = train(form = Purchased ~ ., data = training_set, method = 'svmRadial')
classifier
classifier$bestTune

# Visualising the Training set results
# ElemStatLearn was used in the course for plotting but has been removed from CRAN;
# the plot below only needs base R, so you can skip this library line if it fails.
# Make a fine grid of points, predict buy / not buy at each one, colour the background
# (green = buys, red = doesn't), then draw the real people on top as dots.
library(ElemStatLearn)
set = training_set
X1 = seq(min(set[, 1]) - 1, max(set[, 1]) + 1, by = 0.01)
X2 = seq(min(set[, 2]) - 1, max(set[, 2]) + 1, by = 0.01)
grid_set = expand.grid(X1, X2)
colnames(grid_set) = c('Age', 'EstimatedSalary')
y_grid = predict(classifier, newdata = grid_set)
plot(set[, -3],
     main = 'Kernel SVM (Training set)',
     xlab = 'Age', ylab = 'Estimated Salary',
     xlim = range(X1), ylim = range(X2))
contour(X1, X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid == 1, 'springgreen3', 'tomato'))
points(set, pch = 21, bg = ifelse(set[, 3] == 1, 'green4', 'red3'))

# Visualising the Test set results
# Same chart for the 100 test people the model never saw.
# Dots in a region of the other colour are the model's mistakes.
library(ElemStatLearn)
set = test_set
X1 = seq(min(set[, 1]) - 1, max(set[, 1]) + 1, by = 0.01)
X2 = seq(min(set[, 2]) - 1, max(set[, 2]) + 1, by = 0.01)
grid_set = expand.grid(X1, X2)
colnames(grid_set) = c('Age', 'EstimatedSalary')
y_grid = predict(classifier, newdata = grid_set)
plot(set[, -3], main = 'Kernel SVM (Test set)',
     xlab = 'Age', ylab = 'Estimated Salary',
     xlim = range(X1), ylim = range(X2))
contour(X1, X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid == 1, 'springgreen3', 'tomato'))
points(set, pch = 21, bg = ifelse(set[, 3] == 1, 'green4', 'red3'))