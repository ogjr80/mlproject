# Data Preprocessing Template

# What this script does: it is a short starter skeleton you copy at the
# beginning of a new project. It loads a CSV file and splits the rows into a
# training set (for the model to learn from) and a test set (held back to check it).
# Data: 'Data.csv' (in the course it has Country, Age, Salary and Purchased).
# 'DependentVariable' below is a placeholder: replace it with the name of the
# column you want to predict.
# This is a reusable starting template.

# Importing the dataset
# read.csv loads the file into a "data frame", R's name for a table.
# The CSV must be in your working directory (in RStudio: Session > Set Working Directory).
dataset = read.csv('Data.csv')

# Splitting the dataset into the Training set and Test set
# The install.packages line is commented out because you only need to run it
# once per computer. Remove the '#' the first time, then put it back.
# set.seed(123) fixes R's random number generator, so the "random" split comes
# out the same every time you run the script (like shuffling a deck the same way).
# sample.split gives one TRUE/FALSE per row. SplitRatio = 0.8 is the TRAINING
# share: 80% of rows get TRUE (training), 20% get FALSE (test). It splits on the
# column you want to predict, so both sets keep the same mix of answers.
# subset(dataset, split == TRUE) keeps only the rows marked TRUE.
# install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(dataset$DependentVariable, SplitRatio = 0.8)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# Feature Scaling
# Commented out because many models don't need it. When they do, scale()
# rescales each column so its average is 0 and most values fall between about
# -2 and +2, so big numbers (salary) don't drown out small ones (age).
# Note: scale() only works on number columns, so you usually leave out the
# label/text columns, e.g. training_set[-3] = scale(training_set[-3]).
# training_set = scale(training_set)
# test_set = scale(test_set)