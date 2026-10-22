# Data Preprocessing Template

# What this script does: a reusable starting point for a new project.
# It loads a CSV file, splits the rows into a training set and a test set,
# and shows (commented out) how to put the columns on the same scale.
# Data: 'Data.csv' is a placeholder name. Swap in your own file and column names.
# Python version: none for this template (the clustering notebooks in ../Python start from scratch).

# Importing the dataset
# read.csv reads a comma-separated file into a table R calls a "data frame".
dataset = read.csv('Data.csv')

# Splitting the dataset into the Training set and Test set
# The model learns from the training set and is checked on the test set,
# like studying from one set of questions and sitting an exam on new ones.
# install.packages() only needs to run once per computer, so it is left commented out.
# library() loads the installed package into this session (needed every time).
# set.seed(123) fixes R's random number generator, so the "random" split
# comes out the same every run and your results match the course.
# sample.split keeps 80% of rows (TRUE) for training and 20% (FALSE) for testing,
# with the same mix of answers in both. Replace DependentVariable with your answer column.
# install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(dataset$DependentVariable, SplitRatio = 0.8)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# Feature Scaling
# scale() rescales each column to average 0 and spread 1, so big numbers
# (like salary) don't drown out small ones (like age).
# It is commented out because not every model needs it. Remove the "# " to use it.
# training_set = scale(training_set)
# test_set = scale(test_set)