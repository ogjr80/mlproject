# Kernel PCA

# What this script does: predicts whether a person bought a product after seeing a
# social-network ad, from their age and salary. Kernel PCA first "bends" the data into
# 2 new columns where curved patterns become easier to separate with a straight line,
# then logistic regression is trained on those 2 columns.
# Data: Social_Network_Ads.csv (400 people: Age, EstimatedSalary, Purchased).
# Python version: ../Python/kernel_pca.ipynb (note: the notebook uses the Wine data instead).

# Importing the dataset
# dataset[, 3:5] keeps columns 3 to 5: Age, EstimatedSalary and Purchased.
dataset = read.csv('Social_Network_Ads.csv')
dataset = dataset[, 3:5]

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
# [, 1:2] = columns 1 and 2 (Age and Salary). Salary is in the tens of thousands and
# age is under 100, so scaling puts them on equal footing.
training_set[, 1:2] = scale(training_set[, 1:2])
test_set[, 1:2] = scale(test_set[, 1:2])

# Applying Kernel PCA
# kernlab's kpca(~., data = ...) learns the new columns from the inputs only ([-3] drops Purchased).
# kernel = 'rbfdot' is the RBF ("radial") kernel: it scores how close two people are,
# which lets it capture curved patterns. features = 2 keeps 2 new columns (V1 and V2).
# predict() applies the same transformation to each set, and we copy Purchased back in.
# install.packages('kernlab')
library(kernlab)
kpca = kpca(~., data = training_set[-3], kernel = 'rbfdot', features = 2)
training_set_pca = as.data.frame(predict(kpca, training_set))
training_set_pca$Purchased = training_set$Purchased
test_set_pca = as.data.frame(predict(kpca, test_set))
test_set_pca$Purchased = test_set$Purchased

# Fitting Logistic Regression to the Training set
# glm(..., family = binomial) is logistic regression: it predicts the chance of a yes/no answer.
# Purchased ~ . means "predict Purchased from all other columns" (V1 and V2).
classifier = glm(formula = Purchased ~ .,
                 family = binomial,
                 data = training_set_pca)

# Predicting the Test set results
# type = 'response' returns probabilities between 0 and 1.
# ifelse(prob_pred > 0.5, 1, 0) turns them into 1 (buys) or 0 (doesn't buy).
prob_pred = predict(classifier, type = 'response', newdata = test_set_pca[-3])
y_pred = ifelse(prob_pred > 0.5, 1, 0)

# Making the Confusion Matrix
# Rows = real answer, columns = guess. The diagonal counts the correct guesses.
cm = table(test_set_pca[, 3], y_pred)

# Visualising the Training set results
# Note: the install.packages line below is not commented out, so it reinstalls the package
# every run. ElemStatLearn has also been removed from CRAN, so it may fail; the plot
# itself only needs base R.
# Make a fine grid of points, predict buy / not buy at each one, colour the background
# (green = buys, red = doesn't), then draw the real people on top as dots.
# install.packages('ElemStatLearn')
library(ElemStatLearn)
set = training_set_pca
X1 = seq(min(set[, 1]) - 1, max(set[, 1]) + 1, by = 0.01)
X2 = seq(min(set[, 2]) - 1, max(set[, 2]) + 1, by = 0.01)
grid_set = expand.grid(X1, X2)
colnames(grid_set) = c('V1', 'V2')
prob_set = predict(classifier, type = 'response', newdata = grid_set)
y_grid = ifelse(prob_set > 0.5, 1, 0)
plot(set[, -3],
     main = 'Logistic Regression (Training set)',
     xlab = 'PC1', ylab = 'PC2',
     xlim = range(X1), ylim = range(X2))
contour(X1, X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid == 1, 'springgreen3', 'tomato'))
points(set, pch = 21, bg = ifelse(set[, 3] == 1, 'green4', 'red3'))

# Visualising the Test set results
# Same chart for the 100 test people. (The axis labels say Age / Estimated Salary,
# but the axes are really the two Kernel PCA columns, PC1 and PC2.)
# install.packages('ElemStatLearn')
library(ElemStatLearn)
set = test_set_pca
X1 = seq(min(set[, 1]) - 1, max(set[, 1]) + 1, by = 0.01)
X2 = seq(min(set[, 2]) - 1, max(set[, 2]) + 1, by = 0.01)
grid_set = expand.grid(X1, X2)
colnames(grid_set) = c('V1', 'V2')
prob_set = predict(classifier, type = 'response', newdata = grid_set)
y_grid = ifelse(prob_set > 0.5, 1, 0)
plot(set[, -3],
     main = 'Logistic Regression (Test set)',
     xlab = 'Age', ylab = 'Estimated Salary',
     xlim = range(X1), ylim = range(X2))
contour(X1, X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid == 1, 'springgreen3', 'tomato'))
points(set, pch = 21, bg = ifelse(set[, 3] == 1, 'green4', 'red3'))