# Hierarchical Clustering

# What this script does: splits the same 200 mall shoppers as K-Means into groups.
# It builds a "family tree" by starting with every shopper alone and repeatedly
# merging the two closest groups. The tree (dendrogram) shows that 5 groups is a good choice.
# Data: Mall_Customers.csv (annual income and spending score for each shopper).
# Python version: ../Python/hierarchical_clustering.ipynb

# Importing the dataset
# dataset[4:5] keeps only columns 4 and 5: Annual Income and Spending Score
# (R counts columns from 1). Two columns means we can draw the result on a flat chart.
dataset = read.csv('Mall_Customers.csv')
dataset = dataset[4:5]

# Splitting the dataset into the Training set and Test set
# Not needed: there is no answer column to test against, so these stay commented out.
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

# Using the dendrogram to find the optimal number of clusters
# dist(..., method = 'euclidean') works out the straight-line distance between
# every pair of shoppers, like measuring with a ruler on the chart.
# hclust then does all the merging. method = 'ward.D' merges the two groups that
# keep the groups least spread out, which gives compact, even-sized groups.
# plot() draws the tree. Tall vertical lines mean a merge joined groups that were far apart.
# Cutting across the tallest lines leaves 5 branches, so we choose 5 groups.
dendrogram = hclust(d = dist(dataset, method = 'euclidean'), method = 'ward.D')
plot(dendrogram,
     main = paste('Dendrogram'),
     xlab = 'Customers',
     ylab = 'Euclidean distances')

# Fitting Hierarchical Clustering to the dataset
# Builds the same tree again, then cutree(hc, 5) "cuts" it so that 5 branches remain.
# y_hc is a list of 200 numbers (1 to 5): the group each shopper belongs to.
hc = hclust(d = dist(dataset, method = 'euclidean'), method = 'ward.D')
y_hc = cutree(hc, 5)

# Visualising the clusters
# The 'cluster' package (installed with R) provides clusplot, a ready-made cluster chart.
# clusplot draws every shopper as a point and circles each group:
#   lines = 0 hides lines between groups, shade/color fill the circles,
#   labels = 2 labels points and circles, plotchar = FALSE uses one point symbol,
#   span = TRUE makes each circle just big enough to hold its group.
library(cluster)
clusplot(dataset,
         y_hc,
         lines = 0,
         shade = TRUE,
         color = TRUE,
         labels= 2,
         plotchar = FALSE,
         span = TRUE,
         main = paste('Clusters of customers'),
         xlab = 'Annual Income',
         ylab = 'Spending Score')