# PCA

# What this script does: 178 wines each have 13 chemical measurements and belong to
# one of 3 customer segments. PCA squeezes the 13 columns down to 2 new ones that keep
# as much of the variety as possible. A linear SVM is then trained on those 2 columns,
# and the results are drawn on a flat chart.
# Data: Wine.csv (13 measurements plus Customer_Segment = 1, 2 or 3).
# Python version: ../Python/principal_component_analysis.ipynb (it uses logistic regression instead of SVM).

# Importing the dataset
dataset = read.csv('Wine.csv')

# Splitting the dataset into the Training set and Test set
# set.seed(123) fixes the random split so results match every run.
# About 80% of the wines go to training and 20% to testing.
# install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(dataset$Customer_Segment, SplitRatio = 0.8)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# Feature Scaling
# [-14] = every column except column 14 (the segment). PCA must have scaled data,
# otherwise big-number columns like Proline would dominate the new components.
training_set[-14] = scale(training_set[-14])
test_set[-14] = scale(test_set[-14])

# Applying PCA
# caret's preProcess(method = 'pca', pcaComp = 2) learns, from the training inputs,
# how to mix the 13 columns into 2 new ones (PC1 and PC2). The answer is not used.
# predict(pca, ...) applies that same recipe to a table. The result has the segment
# first, then PC1 and PC2, so [c(2, 3, 1)] reorders it to PC1, PC2, segment.
# install.packages('caret')
library(caret)
# install.packages('e1071')
library(e1071)
pca = preProcess(x = training_set[-14], method = 'pca', pcaComp = 2)
training_set = predict(pca, training_set)
training_set = training_set[c(2, 3, 1)]
test_set = predict(pca, test_set)
test_set = test_set[c(2, 3, 1)]

# Fitting SVM to the Training set
# Customer_Segment ~ . means "predict the segment from all other columns" (PC1 and PC2).
# type = 'C-classification' says we're predicting categories; kernel = 'linear'
# means the dividing lines between segments are straight.
# install.packages('e1071')
library(e1071)
classifier = svm(formula = Customer_Segment ~ .,
                 data = training_set,
                 type = 'C-classification',
                 kernel = 'linear')

# Predicting the Test set results
# [-3] leaves out column 3 (the real segment), so the model only sees PC1 and PC2.
y_pred = predict(classifier, newdata = test_set[-3])

# Making the Confusion Matrix
# Rows = real segment, columns = predicted segment. The diagonal counts correct guesses.
cm = table(test_set[, 3], y_pred)

# Visualising the Training set results
# ElemStatLearn was used in the course for plotting but has been removed from CRAN;
# the plot below only needs base R, so you can skip this library line if it fails.
# The idea: make a fine grid of points covering the chart (seq(..., by = 0.01) and
# expand.grid), ask the model to predict a segment for every grid point, colour the
# background by that prediction, then draw the real wines on top as dots.
# contour() draws the boundary lines between the coloured regions.
library(ElemStatLearn)
set = training_set
X1 = seq(min(set[, 1]) - 1, max(set[, 1]) + 1, by = 0.01)
X2 = seq(min(set[, 2]) - 1, max(set[, 2]) + 1, by = 0.01)
grid_set = expand.grid(X1, X2)
colnames(grid_set) = c('PC1', 'PC2')
y_grid = predict(classifier, newdata = grid_set)
plot(set[, -3],
     main = 'SVM (Training set)',
     xlab = 'PC1', ylab = 'PC2',
     xlim = range(X1), ylim = range(X2))
contour(X1, X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid == 2, 'deepskyblue', ifelse(y_grid == 1, 'springgreen3', 'tomato')))
points(set, pch = 21, bg = ifelse(set[, 3] == 2, 'blue3', ifelse(set[, 3] == 1, 'green4', 'red3')))

# Visualising the Test set results
# Same chart, but the dots are now the test wines the model never saw.
# Dots sitting in a region of a different colour are the model's mistakes.
library(ElemStatLearn)
set = test_set
X1 = seq(min(set[, 1]) - 1, max(set[, 1]) + 1, by = 0.01)
X2 = seq(min(set[, 2]) - 1, max(set[, 2]) + 1, by = 0.01)
grid_set = expand.grid(X1, X2)
colnames(grid_set) = c('PC1', 'PC2')
y_grid = predict(classifier, newdata = grid_set)
plot(set[, -3], main = 'SVM (Test set)',
     xlab = 'PC1', ylab = 'PC2',
     xlim = range(X1), ylim = range(X2))
contour(X1, X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid == 2, 'deepskyblue', ifelse(y_grid == 1, 'springgreen3', 'tomato')))
points(set, pch = 21, bg = ifelse(set[, 3] == 2, 'blue3', ifelse(set[, 3] == 1, 'green4', 'red3')))