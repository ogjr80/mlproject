# Data Preprocessing

# What this script does: it loads a small table of 10 customers, fills in
# missing ages and salaries with the column average, and turns the text
# columns (Country, Purchased) into categories a model can work with.
# Data: 'Data.csv' with columns Country, Age, Salary and Purchased (Yes/No).
# Python version: ../Python/data_preprocessing_tools.ipynb

# Importing the dataset
# read.csv loads the file into a "data frame", R's name for a table.
# Empty cells in number columns become NA ("not available"), R's marker for missing.
dataset = read.csv('Data.csv')

# Taking care of missing data
# dataset$Age means "the Age column of dataset".
# ifelse(test, yes, no) works row by row: if Age is missing (is.na), use the
# column average, otherwise keep the original value.
# na.rm = TRUE tells mean() to skip the missing values while averaging.
dataset$Age = ifelse(is.na(dataset$Age),
                     ave(dataset$Age, FUN = function(x) mean(x, na.rm = TRUE)),
                     dataset$Age)
# The same trick for the Salary column.
dataset$Salary = ifelse(is.na(dataset$Salary),
                        ave(dataset$Salary, FUN = function(x) mean(x, na.rm = TRUE)),
                        dataset$Salary)

# Encoding categorical data
# factor() tells R "this column is a set of categories, not free text or numbers".
# levels lists the possible values; labels gives each one a short code.
# Even though the codes are 1, 2, 3, R treats them as names, not amounts, so a
# model won't think Germany is "3 times" France. (Python needs one-hot encoding
# for this; in R, factor() does that job for you inside models like lm.)
dataset$Country = factor(dataset$Country,
                         levels = c('France', 'Spain', 'Germany'),
                         labels = c(1, 2, 3))
# The answer column: No becomes 0 and Yes becomes 1.
dataset$Purchased = factor(dataset$Purchased,
                           levels = c('No', 'Yes'),
                           labels = c(0, 1))