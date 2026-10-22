# Artificial Neural Network

# What this script does: predicts which of a bank's 10,000 customers will leave
# (Exited = 1) or stay (0), using a small neural network built with the h2o package.
# A neural network is layers of simple "neurons" that pass numbers forward and
# adjust their weights until their guesses match the real answers.
# Data: Churn_Modelling.csv (credit score, country, gender, age, balance, salary, ... and Exited).
# Python version: ../Python/artificial_neural_network.ipynb (it uses TensorFlow/Keras instead of h2o).

# Importing the dataset
# dataset[4:14] keeps columns 4 to 14: the 10 useful inputs plus Exited (now column 11).
# It drops row number, customer ID and surname, which can't predict anything.
dataset = read.csv('Churn_Modelling.csv')
dataset = dataset[4:14]

# Encoding the categorical variables as factors
# The network can only do maths on numbers, so the two word columns become numbers.
# factor(..., levels, labels) maps each word to a code (France = 1, Spain = 2, Germany = 3;
# Female = 1, Male = 2), and as.numeric() turns those codes into plain numbers.
dataset$Geography = as.numeric(factor(dataset$Geography,
                                      levels = c('France', 'Spain', 'Germany'),
                                      labels = c(1, 2, 3)))
dataset$Gender = as.numeric(factor(dataset$Gender,
                                   levels = c('Female', 'Male'),
                                   labels = c(1, 2)))

# Splitting the dataset into the Training set and Test set
# set.seed(123) fixes the random split so results match every run.
# 80% of customers (8,000) for training, 20% (2,000) for testing.
# install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(dataset$Exited, SplitRatio = 0.8)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# Feature Scaling
# [-11] means "every column except column 11" (Exited, the answer, which stays 0/1).
# scale() rescales the inputs to average 0 and spread 1. Neural networks learn much
# better when balance (~100,000) and HasCrCard (0 or 1) are on similar scales.
training_set[-11] = scale(training_set[-11])
test_set[-11] = scale(test_set[-11])

# Fitting ANN to the Training set
# h2o is a fast machine-learning engine that runs as a separate program on your computer.
# h2o.init(nthreads = -1) starts it and lets it use all your CPU cores.
# as.h2o() copies an R table into h2o's own format, which h2o functions need.
# h2o.deeplearning settings:
#   y = 'Exited': the column to predict (all other columns are used as inputs).
#   activation = 'Rectifier': ReLU neurons (pass positive values, block negatives).
#   hidden = c(5,5): two hidden layers with 5 neurons each.
#   epochs = 100: go through the whole training set 100 times.
#   train_samples_per_iteration = -2: let h2o pick the batch size automatically.
# install.packages('h2o')
library(h2o)
h2o.init(nthreads = -1)
model = h2o.deeplearning(y = 'Exited',
                         training_frame = as.h2o(training_set),
                         activation = 'Rectifier',
                         hidden = c(5,5),
                         epochs = 100,
                         train_samples_per_iteration = -2)

# Predicting the Test set results
# h2o.predict gives a number between 0 and 1 for each test customer: the chance they leave.
# (y_pred > 0.5) turns it into TRUE (will leave) / FALSE (will stay).
# as.vector() copies the answers back from h2o into a normal R list.
y_pred = h2o.predict(model, newdata = as.h2o(test_set[-11]))
y_pred = (y_pred > 0.5)
y_pred = as.vector(y_pred)

# Making the Confusion Matrix
# table() counts right and wrong guesses: rows = real answer, columns = guess.
# The diagonal holds the correct guesses (around 86% accuracy is typical here).
cm = table(test_set[, 11], y_pred)

# When you are finished, h2o.shutdown() stops the h2o engine and frees its memory.
# It is commented out so you don't shut it down by accident while still experimenting.
# h2o.shutdown()