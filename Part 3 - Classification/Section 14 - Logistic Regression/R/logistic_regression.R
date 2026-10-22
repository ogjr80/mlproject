# Logistic Regression

# What this script does: it predicts whether a person will buy a product after
# seeing an ad, from their age and salary. Logistic regression works out a
# probability between 0 and 1 (an S-shaped curve); above 50% means "will buy".
# It is checked with a confusion matrix and coloured maps of its decisions.
# Data: 'Social_Network_Ads.csv' with 400 people. Columns used: Age,
# EstimatedSalary and Purchased (1 = bought, 0 = didn't).
# Python version: ../Python/logistic_regression.ipynb

# Importing the dataset
# dataset[3:5] keeps columns 3 to 5 (Age, EstimatedSalary, Purchased) and
# drops User.ID and Gender.
dataset = read.csv('Social_Network_Ads.csv')
dataset = dataset[3:5]

# Encoding the target feature as factor
# factor() marks Purchased as a category (0 or 1), not a number. This tells
# the model to classify (pick a group) rather than predict an amount.
dataset$Purchased = factor(dataset$Purchased, levels = c(0, 1))

# Splitting the dataset into the Training set and Test set
# install.packages is commented out: run it once per computer, then leave it off.
# set.seed(123) fixes the random shuffle so you get the same split every run.
# sample.split marks each row TRUE (training) or FALSE (test). SplitRatio = 0.75
# is the TRAINING share: 300 people to learn from, 100 held back for testing.
# It splits on Purchased, so both sets keep the same mix of buyers and non-buyers.
# subset() then keeps only the rows with the chosen mark.
# install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(dataset$Purchased, SplitRatio = 0.75)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# Feature Scaling
# Age is about 18-60 but salary is 15,000-150,000. scale() puts both on the same
# footing: average 0, most values between about -2 and +2.
# [-3] means "every column except the 3rd": Purchased is the answer (a factor),
# so it must not be scaled. Note: unlike the Python version, the test set is
# scaled with its own average and spread rather than the training set's.
training_set[-3] = scale(training_set[-3])
test_set[-3] = scale(test_set[-3])

# Fitting Logistic Regression to the Training set
# glm = "generalised linear model". Purchased ~ . reads "explain Purchased
# using every other column" (Age and EstimatedSalary).
# family = binomial makes it logistic regression: a yes/no answer, with the
# result squeezed through an S-curve into a probability between 0 and 1.
classifier = glm(formula = Purchased ~ .,
                 family = binomial,
                 data = training_set)

# Predicting the Test set results
# type = 'response' asks for probabilities (e.g. 0.83 = 83% chance of buying)
# instead of the raw score before the S-curve. test_set[-3] hides the answer.
# ifelse(prob_pred > 0.5, 1, 0) turns each probability into a firm answer:
# more than 50% means 1 (will buy), otherwise 0.
prob_pred = predict(classifier, type = 'response', newdata = test_set[-3])
y_pred = ifelse(prob_pred > 0.5, 1, 0)

# Making the Confusion Matrix
# table(real answers, predictions) counts every combination, like a tally chart.
# Rows = what really happened (0/1), columns = what the model said. Because
# y_pred > 0.5 is a TRUE/FALSE test, the columns are labelled FALSE (0) and
# TRUE (1). The diagonal (0-FALSE, 1-TRUE) holds the correct guesses.
cm = table(test_set[, 3], y_pred > 0.5)

# Visualising the Training set results
# This draws a map of the model's decisions. (ElemStatLearn has been removed
# from CRAN; everything below is base R, so you can skip that line if it fails.)
# X1 and X2 run through every age and salary value from just below the smallest
# to just above the largest, in tiny steps of 0.01 (on the scaled values).
# expand.grid pairs every X1 with every X2: a fine grid of imaginary people that
# covers the whole chart, like pixels on a screen. colnames gives the grid the
# same column names as the training data so predict() recognises them.
# The model gives each pixel a probability, which ifelse turns into 0 or 1.
library(ElemStatLearn)
set = training_set
X1 = seq(min(set[, 1]) - 1, max(set[, 1]) + 1, by = 0.01)
X2 = seq(min(set[, 2]) - 1, max(set[, 2]) + 1, by = 0.01)
grid_set = expand.grid(X1, X2)
colnames(grid_set) = c('Age', 'EstimatedSalary')
prob_set = predict(classifier, type = 'response', newdata = grid_set)
y_grid = ifelse(prob_set > 0.5, 1, 0)
# plot() sets up the chart; set[, -3] is Age and Salary without the answer.
# contour() draws the boundary line where the predictions switch from 0 to 1
# (a straight line, because logistic regression is a linear model).
# The first points() colours each pixel: green = predicted buyer, red = not.
# The second points() draws the real people on top (pch = 21 is a filled
# circle), coloured by what they actually did.
plot(set[, -3],
     main = 'Logistic Regression (Training set)',
     xlab = 'Age', ylab = 'Estimated Salary',
     xlim = range(X1), ylim = range(X2))
contour(X1, X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid == 1, 'springgreen3', 'tomato'))
points(set, pch = 21, bg = ifelse(set[, 3] == 1, 'green4', 'red3'))

# Visualising the Test set results
# The same map, but with the 100 unseen test people as the dots. Dots sitting
# in the wrong colour zone are the mistakes counted in the confusion matrix.
library(ElemStatLearn)
set = test_set
X1 = seq(min(set[, 1]) - 1, max(set[, 1]) + 1, by = 0.01)
X2 = seq(min(set[, 2]) - 1, max(set[, 2]) + 1, by = 0.01)
grid_set = expand.grid(X1, X2)
colnames(grid_set) = c('Age', 'EstimatedSalary')
prob_set = predict(classifier, type = 'response', newdata = grid_set)
y_grid = ifelse(prob_set > 0.5, 1, 0)
plot(set[, -3],
     main = 'Logistic Regression (Test set)',
     xlab = 'Age', ylab = 'Estimated Salary',
     xlim = range(X1), ylim = range(X2))
contour(X1, X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid == 1, 'springgreen3', 'tomato'))
points(set, pch = 21, bg = ifelse(set[, 3] == 1, 'green4', 'red3'))