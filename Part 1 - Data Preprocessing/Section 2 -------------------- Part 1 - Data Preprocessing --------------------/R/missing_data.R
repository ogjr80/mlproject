# Data Preprocessing

# What this script does: it loads a small table of 10 customers and fills in
# the gaps where a value is missing, using the average of that column.
# Data: 'Data.csv' with columns Country, Age, Salary and Purchased (Yes/No).
# Age and Salary each have one missing value.
# Python version: ../Python/data_preprocessing_tools.ipynb

# Importing the dataset
# read.csv loads the file into a "data frame", R's name for a table.
# Empty cells in number columns become NA ("not available"), R's marker for missing.
dataset = read.csv('Data.csv')

# Taking care of missing data
# dataset$Age means "the Age column of dataset".
# ifelse(test, yes, no) works row by row: if Age is missing (is.na), use the
# column average, otherwise keep the original value.
# ave(..., FUN = mean(x, na.rm = TRUE)) works out the average while skipping
# the missing values (na.rm = "NA remove"); without it the average would be NA too.
# Why the average? It is a sensible guess that doesn't throw away the whole row.
dataset$Age = ifelse(is.na(dataset$Age),
                     ave(dataset$Age, FUN = function(x) mean(x, na.rm = TRUE)),
                     dataset$Age)
# The same trick for the Salary column.
dataset$Salary = ifelse(is.na(dataset$Salary),
                        ave(dataset$Salary, FUN = function(x) mean(x, na.rm = TRUE)),
                        dataset$Salary)