# Naive Bayes

# What this script does: it predicts whether a person will buy a product after
# seeing an ad, from their age and salary, with Naive Bayes. Like a detective
# weighing clues, it asks how typical this age and salary are among buyers versus
# non-buyers (and how common buyers are overall), then picks the likelier answer.
# Note: some headings and chart titles below say 'SVM'. That is a copy-paste
# leftover; the model really is Naive Bayes.
# Data: 'Social_Network_Ads.csv' with 400 people who saw an ad. Columns used:
# Age, EstimatedSalary and Purchased (1 = bought, 0 = didn't).
# Python version: ../Python/naive_bayes.ipynb

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
# Why: Naive Bayes doesn't strictly need it, but it keeps the code the same
# as the other classification scripts and makes the chart grid work.
# [-3] means "every column except the 3rd": Purchased is the answer (a factor),
# so it must not be scaled. Note: unlike the Python version, the test set is
# scaled with its own average and spread rather than the training set's.
training_set[-3] = scale(training_set[-3])
test_set[-3] = scale(test_set[-3])

# Fitting SVM to the Training set
# e1071 also provides naiveBayes(). Instead of a formula it takes the inputs
# and the answer separately: x = training_set[-3] (Age and Salary) and
# y = training_set$Purchased (the answer). y must be a factor, which is why we
# encoded Purchased with factor() at the start.
# install.packages('e1071')
library(e1071)
classifier = naiveBayes(x = training_set[-3],
                        y = training_set$Purchased)

# Predicting the Test set results
# test_set[-3] passes only Age and Salary: the model must guess Purchased itself.
y_pred = predict(classifier, newdata = test_set[-3])

# Making the Confusion Matrix
# table(real answers, predictions) counts every combination, like a tally chart.
# Rows = what really happened, columns = what the model said. The two numbers
# on the diagonal (0-0 and 1-1) are correct guesses; the other two are mistakes.
cm = table(test_set[, 3], y_pred)

# Visualising the Training set results
# This draws a map of the model's decisions. (ElemStatLearn has been removed
# from CRAN; everything below is base R, so you can skip that line if it fails.)
# X1 and X2 run through every age and salary value from just below the smallest
# to just above the largest, in tiny steps of 0.01 (on the scaled values).
# expand.grid pairs every X1 with every X2: a fine grid of imaginary people that
# covers the whole chart, like pixels on a screen. colnames gives the grid the
# same column names as the training data so the model recognises them.
# The model then predicts buy / not buy for every pixel.
library(ElemStatLearn)
set = training_set
X1 = seq(min(set[, 1]) - 1, max(set[, 1]) + 1, by = 0.01)
X2 = seq(min(set[, 2]) - 1, max(set[, 2]) + 1, by = 0.01)
grid_set = expand.grid(X1, X2)
colnames(grid_set) = c('Age', 'EstimatedSalary')
y_grid = predict(classifier, newdata = grid_set)
# plot() sets up the chart; set[, -3] is Age and Salary without the answer.
# contour() draws the boundary line where the predictions switch from 0 to 1.
# The first points() colours each pixel: green = predicted buyer, red = not.
# The second points() draws the real people on top (pch = 21 is a filled
# circle), coloured by what they actually did.
plot(set[, -3],
     main = 'SVM (Training set)',
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
y_grid = predict(classifier, newdata = grid_set)
plot(set[, -3], main = 'SVM (Test set)',
     xlab = 'Age', ylab = 'Estimated Salary',
     xlim = range(X1), ylim = range(X2))
contour(X1, X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid == 1, 'springgreen3', 'tomato'))
points(set, pch = 21, bg = ifelse(set[, 3] == 1, 'green4', 'red3'))