# K-Means Clustering

# What this script does: splits 200 mall shoppers into groups of similar people.
# It first uses the "elbow method" to pick how many groups (K) to make,
# then runs K-Means with K = 5 and draws the groups on a chart.
# This is unsupervised learning: there is no answer column, the computer finds the groups itself.
# Data: Mall_Customers.csv (annual income and spending score for each shopper).
# Python version: ../Python/k_means_clustering.ipynb

# Importing the dataset
# dataset[4:5] keeps only columns 4 and 5: Annual Income and Spending Score.
# R counts columns from 1 (Python counts from 0, which is why the notebook says 3 and 4).
# With just two columns we can draw the groups on a flat chart.
dataset = read.csv('Mall_Customers.csv')
dataset = dataset[4:5]

# Splitting the dataset into the Training set and Test set
# Not needed here: with no answer column there is nothing to test against,
# so these template lines stay commented out.
# install.packages('caTools')
# library(caTools)
# set.seed(123)
# split = sample.split(dataset$DependentVariable, SplitRatio = 0.8)
# training_set = subset(dataset, split == TRUE)
# test_set = subset(dataset, split == FALSE)

# Feature Scaling
# Also skipped: income and spending score are already on similar ranges.
# training_set = scale(training_set)
# test_set = scale(test_set)

# Using the elbow method to find the optimal number of clusters
# K-Means needs K (the number of groups) in advance, so we try K = 1 to 10.
# set.seed(6) fixes the random starting points, so the chart looks the same every run.
# wcss = vector() makes an empty list to fill in.
# kmeans(dataset, i)$withinss gives, for each group, how spread out its members are
# around the centre. sum() adds them up: this is WCSS (within-cluster sum of squares).
# Smaller WCSS = tighter groups. We look for the "elbow" where the line stops dropping
# fast. Here that is at 5. type = 'b' draws both dots and lines.
set.seed(6)
wcss = vector()
for (i in 1:10) wcss[i] = sum(kmeans(dataset, i)$withinss)
plot(1:10,
     wcss,
     type = 'b',
     main = paste('The Elbow Method'),
     xlab = 'Number of clusters',
     ylab = 'WCSS')

# Fitting K-Means to the dataset
# centers = 5 uses the K read off the elbow chart.
# K-Means drops 5 random "flags", moves each shopper to the nearest flag,
# moves each flag to the middle of its shoppers, and repeats until nothing changes.
# Extra options exist: iter.max (most rounds allowed, default 10) and
# nstart (how many random starts to try and keep the best, default 1).
# set.seed(29) fixes the random start so the groups come out the same every run.
# kmeans$cluster is a list of 200 numbers (1 to 5): the group each shopper landed in.
# (Note: this line reuses the name "kmeans" for the result. It works, but a
# different name would be clearer.)
set.seed(29)
kmeans = kmeans(x = dataset, centers = 5)
y_kmeans = kmeans$cluster

# Visualising the clusters
# The 'cluster' package (installed with R) provides clusplot, a ready-made cluster chart.
# clusplot draws every shopper as a point and circles each group:
#   lines = 0 hides lines between groups, shade/color fill the circles,
#   labels = 2 labels points and circles, plotchar = FALSE uses one point symbol,
#   span = TRUE makes each circle just big enough to hold its group.
# Note: clusplot works on a squashed view of the data, so the axes are
# "components" rather than raw income and score, even though the labels say so.
library(cluster)
clusplot(dataset,
         y_kmeans,
         lines = 0,
         shade = TRUE,
         color = TRUE,
         labels = 2,
         plotchar = FALSE,
         span = TRUE,
         main = paste('Clusters of customers'),
         xlab = 'Annual Income',
         ylab = 'Spending Score')