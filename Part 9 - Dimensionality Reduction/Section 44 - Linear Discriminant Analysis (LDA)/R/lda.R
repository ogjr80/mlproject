# LDA

# What this script does: the same 178 wines (13 measurements, 3 customer segments) as PCA.
# LDA (Linear Discriminant Analysis) also squeezes 13 columns down to 2, but unlike PCA
# it looks at the segment labels and picks the 2 directions that best separate the segments.
# A linear SVM is then trained on those 2 columns and the results are drawn on a chart.
# Data: Wine.csv (13 measurements plus Customer_Segment = 1, 2 or 3).
# Python version: ../Python/linear_discriminant_analysis.ipynb (it uses logistic regression instead of SVM).

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
# [-14] = every column except column 14 (the segment). Puts all 13 measurements on the
# same scale so no column wins just because its numbers are bigger.
training_set[-14] = scale(training_set[-14])
test_set[-14] = scale(test_set[-14])

# Applying LDA
# MASS comes with R. lda(Customer_Segment ~ .) learns the best 2 "viewing angles"
# (LD1 and LD2) using the segment labels. With 3 segments, LDA gives at most 2 of them.
# predict() returns several things (class, posterior probabilities, x.LD1, x.LD2);
# as.data.frame() flattens them into one table. [c(5, 6, 1)] keeps x.LD1, x.LD2 and
# the class (the segment), in that order.
library(MASS)
lda = lda(formula = Customer_Segment ~ ., data = training_set)
training_set = as.data.frame(predict(lda, training_set))
training_set = training_set[c(5, 6, 1)]
test_set = as.data.frame(predict(lda, test_set))
test_set = test_set[c(5, 6, 1)]

# Fitting SVM to the Training set
# The segment column is now called "class", so the formula is class ~ . (predict class
# from LD1 and LD2). kernel = 'linear' gives straight dividing lines.
# install.packages('e1071')
library(e1071)
classifier = svm(formula = class ~ .,
                 data = training_set,
                 type = 'C-classification',
                 kernel = 'linear')

# Predicting the Test set results
# [-3] leaves out column 3 (the real segment), so the model only sees LD1 and LD2.
y_pred = predict(classifier, newdata = test_set[-3])

# Making the Confusion Matrix
# Rows = real segment, columns = predicted segment. With LDA all test wines are
# usually classified correctly (everything on the diagonal).
cm = table(test_set[, 3], y_pred)

# Visualising the Training set results
# ElemStatLearn was used in the course for plotting but has been removed from CRAN;
# the plot below only needs base R, so you can skip this library line if it fails.
# Make a fine grid of points over the chart, predict a segment at each one, colour the
# background by that prediction, then draw the real wines on top as dots.
# The grid columns are named x.LD1 / x.LD2 to match the names the model was trained on.
library(ElemStatLearn)
set = training_set
X1 = seq(min(set[, 1]) - 1, max(set[, 1]) + 1, by = 0.01)
X2 = seq(min(set[, 2]) - 1, max(set[, 2]) + 1, by = 0.01)
grid_set = expand.grid(X1, X2)
colnames(grid_set) = c('x.LD1', 'x.LD2')
y_grid = predict(classifier, newdata = grid_set)
plot(set[, -3],
     main = 'SVM (Training set)',
     xlab = 'LD1', ylab = 'LD2',
     xlim = range(X1), ylim = range(X2))
contour(X1, X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid == 2, 'deepskyblue', ifelse(y_grid == 1, 'springgreen3', 'tomato')))
points(set, pch = 21, bg = ifelse(set[, 3] == 2, 'blue3', ifelse(set[, 3] == 1, 'green4', 'red3')))

# Visualising the Test set results
# Same chart for the test wines the model never saw.
library(ElemStatLearn)
set = test_set
X1 = seq(min(set[, 1]) - 1, max(set[, 1]) + 1, by = 0.01)
X2 = seq(min(set[, 2]) - 1, max(set[, 2]) + 1, by = 0.01)
grid_set = expand.grid(X1, X2)
colnames(grid_set) = c('x.LD1', 'x.LD2')
y_grid = predict(classifier, newdata = grid_set)
plot(set[, -3], main = 'SVM (Test set)',
     xlab = 'LD1', ylab = 'LD2',
     xlim = range(X1), ylim = range(X2))
contour(X1, X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid == 2, 'deepskyblue', ifelse(y_grid == 1, 'springgreen3', 'tomato')))
points(set, pch = 21, bg = ifelse(set[, 3] == 2, 'blue3', ifelse(set[, 3] == 1, 'green4', 'red3')))