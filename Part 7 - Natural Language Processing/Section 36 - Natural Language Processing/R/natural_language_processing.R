# Natural Language Processing

# What this script does: teaches a computer to read 1,000 restaurant reviews and guess
# whether each one is positive (Liked = 1) or negative (Liked = 0). It cleans the text,
# turns each review into word counts ("bag of words"), then trains a Random Forest on them.
# Data: Restaurant_Reviews.tsv (one short review per row plus the Liked label).
# Python version: ../Python/natural_language_processing.ipynb (it uses Naive Bayes instead).

# Importing the dataset
# read.delim reads a tab-separated (.tsv) file. Tabs are used because reviews contain commas.
# quote = '' ignores quote marks inside the text, so they don't break the reading.
# stringsAsFactors = FALSE keeps the reviews as plain text instead of categories.
dataset_original = read.delim('Restaurant_Reviews.tsv', quote = '', stringsAsFactors = FALSE)

# Cleaning the texts
# tm is R's text-mining package; SnowballC provides word stemming. Install them once.
# VCorpus(VectorSource(...)) puts the 1,000 reviews into a "corpus": a collection of documents.
# tm_map(corpus, step) applies one cleaning step to every review:
#   content_transformer(tolower): make everything lower case, so "Good" = "good".
#   removeNumbers / removePunctuation: drop digits and punctuation.
#   removeWords, stopwords(): drop common filler words like "the", "and", "is".
#     (This list also contains "not", so "not good" loses its "not". A known weakness.)
#   stemDocument: cut words to their root, so "loved" and "loving" both become "love".
#   stripWhitespace: tidy up the extra spaces left behind.
# install.packages('tm')
# install.packages('SnowballC')
library(tm)
library(SnowballC)
corpus = VCorpus(VectorSource(dataset_original$Review))
corpus = tm_map(corpus, content_transformer(tolower))
corpus = tm_map(corpus, removeNumbers)
corpus = tm_map(corpus, removePunctuation)
corpus = tm_map(corpus, removeWords, stopwords())
corpus = tm_map(corpus, stemDocument)
corpus = tm_map(corpus, stripWhitespace)

# Creating the Bag of Words model
# DocumentTermMatrix makes a big table: one row per review, one column per word,
# each cell counting how often that word appears in that review. Word order is ignored.
# removeSparseTerms(dtm, 0.999) drops words that are missing from more than 99.9% of
# reviews (words used only once), leaving about 691 word columns.
# as.data.frame(as.matrix(dtm)) turns it into an ordinary table, and we add the Liked label
# as the last column (column 692).
dtm = DocumentTermMatrix(corpus)
dtm = removeSparseTerms(dtm, 0.999)
dataset = as.data.frame(as.matrix(dtm))
dataset$Liked = dataset_original$Liked

# Importing the dataset
# NOTE: these two lines were left over from the classification template. They would replace
# the bag-of-words table with a different file and break the rest of the script,
# so they are commented out.
# dataset = read.csv('Social_Network_Ads.csv')
# dataset = dataset[3:5]

# Encoding the target feature as factor
# factor() tells R that Liked is a category (0 or 1), not a number to do maths on.
# This makes the Random Forest do classification instead of regression.
dataset$Liked = factor(dataset$Liked, levels = c(0, 1))

# Splitting the dataset into the Training set and Test set
# set.seed(123) fixes the random split so results match every run.
# sample.split puts 80% of reviews (800) in training and 20% (200) in the test set,
# keeping the same mix of liked / not liked in both.
# install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(dataset$Liked, SplitRatio = 0.8)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# Fitting Random Forest Classification to the Training set
# A Random Forest is a team of decision trees that vote. ntree = 10 uses 10 trees.
# x = training_set[-692] means "every column except column 692" (the word counts),
# and y is the Liked label we want to predict.
# install.packages('randomForest')
library(randomForest)
classifier = randomForest(x = training_set[-692],
                          y = training_set$Liked,
                          ntree = 10)

# Predicting the Test set results
# Guesses liked / not liked for the 200 test reviews (again leaving out the answer column).
y_pred = predict(classifier, newdata = test_set[-692])

# Making the Confusion Matrix
# table() counts right and wrong guesses: rows = real answer, columns = guess.
# The diagonal (top-left and bottom-right) holds the correct guesses.
cm = table(test_set[, 692], y_pred)