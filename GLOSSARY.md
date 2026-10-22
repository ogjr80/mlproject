# Glossary

Plain-English meanings of the words used in this course's notebooks, grouped by topic and sorted A–Z within each group.

**How to use it:** when a notebook uses a word you don't know, look it up here. Each entry ends with a link to the notebook where the word first comes up. Read that part of the notebook to see the idea in action.

**Jump to:** [Data & preparation](#data--preparation) · [Training & testing](#training--testing) · [Regression](#regression) · [Classification](#classification) · [Measuring results](#measuring-results) · [Clustering](#clustering) · [Association rules](#association-rules) · [Reinforcement learning](#reinforcement-learning) · [Text (NLP)](#text-nlp) · [Neural networks](#neural-networks) · [Dimensionality reduction](#dimensionality-reduction) · [Tuning & boosting](#tuning--boosting) · [Tools & libraries](#tools--libraries)

---

## Data & preparation

**Categorical data** — Data made of labels or categories (like a country name or Yes/No) rather than measurements. Models only do maths on numbers, so categories have to be turned into numbers first. *Example:* the Country column in `Data.csv`. See: [data_preprocessing_tools](Part%201%20-%20Data%20Preprocessing/Section%202%20--------------------%20Part%201%20-%20Data%20Preprocessing%20--------------------/Python/data_preprocessing_tools.ipynb)

**Data leakage** — When information from the test set sneaks into training, so the model looks better than it really is. This is why the scaler (and PCA) are fitted on the training set only, then just *applied* to the test set. See: [data_preprocessing_tools](Part%201%20-%20Data%20Preprocessing/Section%202%20--------------------%20Part%201%20-%20Data%20Preprocessing%20--------------------/Python/data_preprocessing_tools.ipynb)

**Dataset** — The table of data you learn from. Each row is one example (a customer, a wine, a review) and each column is one piece of information about it. See: [data_preprocessing_template](Part%201%20-%20Data%20Preprocessing/Section%202%20--------------------%20Part%201%20-%20Data%20Preprocessing%20--------------------/Python/data_preprocessing_template.ipynb)

**Dependent variable** — The column you want to predict, usually called **y**. It "depends" on the inputs. Also called the *target* or *label*. *Example:* Purchased (Yes/No). See: [data_preprocessing_tools](Part%201%20-%20Data%20Preprocessing/Section%202%20--------------------%20Part%201%20-%20Data%20Preprocessing%20--------------------/Python/data_preprocessing_tools.ipynb)

**Dummy variable** — One of the new 0/1 columns created by one-hot encoding. *Example:* the "France", "Germany" and "Spain" columns. See: [data_preprocessing_tools](Part%201%20-%20Data%20Preprocessing/Section%202%20--------------------%20Part%201%20-%20Data%20Preprocessing%20--------------------/Python/data_preprocessing_tools.ipynb)

**Dummy variable trap** — With one-hot encoding, the dummy columns always add up to 1, so one of them is redundant (you can work it out from the others). Normally you drop one, but scikit-learn's `LinearRegression` handles this for you. See: [multiple_linear_regression](Part%202%20-%20Regression/Section%205%20-%20Multiple%20Linear%20Regression/Python/multiple_linear_regression.ipynb)

**Feature** — One input column that the model uses to make its guess. Also called an *independent variable*. *Example:* Age and Estimated Salary. See: [data_preprocessing_tools](Part%201%20-%20Data%20Preprocessing/Section%202%20--------------------%20Part%201%20-%20Data%20Preprocessing%20--------------------/Python/data_preprocessing_tools.ipynb)

**Feature scaling** — Putting all number columns on a similar range so a column with big numbers (salary in the tens of thousands) doesn't drown out one with small numbers (age). Needed for models that measure distances or add up "weight × input"; not needed for decision trees. See: [data_preprocessing_tools](Part%201%20-%20Data%20Preprocessing/Section%202%20--------------------%20Part%201%20-%20Data%20Preprocessing%20--------------------/Python/data_preprocessing_tools.ipynb)

**fit / transform / fit_transform** — `fit` *learns* something from data (for example the average of each column), `transform` *applies* it, and `fit_transform` does both. Use `fit_transform` on the training set and only `transform` on the test set. See: [data_preprocessing_tools](Part%201%20-%20Data%20Preprocessing/Section%202%20--------------------%20Part%201%20-%20Data%20Preprocessing%20--------------------/Python/data_preprocessing_tools.ipynb)

**Imputation** — Filling in missing values instead of deleting the rows. The course fills each gap with the column's average (`SimpleImputer`, `strategy='mean'`). *Example:* the missing salary became 63,777.78. See: [data_preprocessing_tools](Part%201%20-%20Data%20Preprocessing/Section%202%20--------------------%20Part%201%20-%20Data%20Preprocessing%20--------------------/Python/data_preprocessing_tools.ipynb)

**Independent variable** — Another name for an input column (a feature), usually grouped together as **X**. See: [data_preprocessing_tools](Part%201%20-%20Data%20Preprocessing/Section%202%20--------------------%20Part%201%20-%20Data%20Preprocessing%20--------------------/Python/data_preprocessing_tools.ipynb)

**Inverse transform** — Undoing a scaling step to get back to real units, such as dollars or years. *Example:* SVR's prediction is turned back into dollars with `sc_y.inverse_transform`. See: [support_vector_regression](Part%202%20-%20Regression/Section%207%20-%20Support%20Vector%20Regression%20%28SVR%29/Python/support_vector_regression.ipynb)

**Label** — The answer for each row (the value in the y column). "Labelled data" means data that comes with the right answers. See: [data_preprocessing_tools](Part%201%20-%20Data%20Preprocessing/Section%202%20--------------------%20Part%201%20-%20Data%20Preprocessing%20--------------------/Python/data_preprocessing_tools.ipynb)

**Label encoding** — Turning each category into a single number, such as No → 0 and Yes → 1 (alphabetical order). Safe for two options. With three or more it would suggest a false order, so use one-hot encoding instead. See: [data_preprocessing_tools](Part%201%20-%20Data%20Preprocessing/Section%202%20--------------------%20Part%201%20-%20Data%20Preprocessing%20--------------------/Python/data_preprocessing_tools.ipynb)

**Missing data** — Empty cells in the table. Python shows them as `nan` ("not a number"). Most models crash on them, so they are filled in (see *Imputation*). See: [data_preprocessing_tools](Part%201%20-%20Data%20Preprocessing/Section%202%20--------------------%20Part%201%20-%20Data%20Preprocessing%20--------------------/Python/data_preprocessing_tools.ipynb)

**One-hot encoding** — Replacing a category column with one 0/1 column per category, so no category looks "bigger" than another. *Example:* France → `1, 0, 0`, Germany → `0, 1, 0`, Spain → `0, 0, 1`. See: [data_preprocessing_tools](Part%201%20-%20Data%20Preprocessing/Section%202%20--------------------%20Part%201%20-%20Data%20Preprocessing%20--------------------/Python/data_preprocessing_tools.ipynb)

**Outlier** — A value far away from all the others. *Example:* the CEO's $1,000,000 salary, which SVR does not stretch its curve to reach. See: [support_vector_regression](Part%202%20-%20Regression/Section%207%20-%20Support%20Vector%20Regression%20%28SVR%29/Python/support_vector_regression.ipynb)

**Standardisation** — The kind of feature scaling used in this course (`StandardScaler`): *(value − average) ÷ spread*. After it, each column has an average of 0 and most values fall roughly between −2 and +2 (a few can reach about ±3). See: [data_preprocessing_tools](Part%201%20-%20Data%20Preprocessing/Section%202%20--------------------%20Part%201%20-%20Data%20Preprocessing%20--------------------/Python/data_preprocessing_tools.ipynb)

**X and y** — The usual names for the inputs (X, a table of features) and the answer column (y). See: [data_preprocessing_template](Part%201%20-%20Data%20Preprocessing/Section%202%20--------------------%20Part%201%20-%20Data%20Preprocessing%20--------------------/Python/data_preprocessing_template.ipynb)

## Training & testing

**Baseline** — A simple model or score used as a yardstick for better ones. *Example:* the straight-line model in the polynomial notebook, or the 90.33% default score in grid search. See: [polynomial_regression](Part%202%20-%20Regression/Section%206%20-%20Polynomial%20Regression/Python/polynomial_regression.ipynb)

**Model** — The thing that learns a pattern from data and then makes predictions. In code it is an object such as `LinearRegression()` or `SVC()`. See: [simple_linear_regression](Part%202%20-%20Regression/Section%204%20-%20Simple%20Linear%20Regression/Python/simple_linear_regression.ipynb)

**Overfitting** — When a model memorises the training data, including its quirks and noise, instead of learning the general pattern. It looks great on training data and worse on new data. *Example:* the unlimited decision tree's little "islands" around single dots. See: [polynomial_regression](Part%202%20-%20Regression/Section%206%20-%20Polynomial%20Regression/Python/polynomial_regression.ipynb)

**Predict** — Using a trained model to guess the answer for new rows. Note the double brackets in `predict([[6.5]])`: the model always expects a table, even for one row. See: [simple_linear_regression](Part%202%20-%20Regression/Section%204%20-%20Simple%20Linear%20Regression/Python/simple_linear_regression.ipynb)

**random_state** — A fixed "seed" number that makes shuffling and other random steps come out the same every time you run the code, so your results match the notebook's. See: [data_preprocessing_template](Part%201%20-%20Data%20Preprocessing/Section%202%20--------------------%20Part%201%20-%20Data%20Preprocessing%20--------------------/Python/data_preprocessing_template.ipynb)

**Supervised learning** — Learning from examples that come with the right answers (a y column). Regression and classification are supervised. See: [simple_linear_regression](Part%202%20-%20Regression/Section%204%20-%20Simple%20Linear%20Regression/Python/simple_linear_regression.ipynb)

**Test set** — The rows kept aside and *not* used for learning. They act as a final exam with new questions, to check the model really learned the pattern. See: [data_preprocessing_template](Part%201%20-%20Data%20Preprocessing/Section%202%20--------------------%20Part%201%20-%20Data%20Preprocessing%20--------------------/Python/data_preprocessing_template.ipynb)

**Train/test split** — Dividing the data into a training set and a test set, for example 75% / 25% (`train_test_split`). See: [data_preprocessing_template](Part%201%20-%20Data%20Preprocessing/Section%202%20--------------------%20Part%201%20-%20Data%20Preprocessing%20--------------------/Python/data_preprocessing_template.ipynb)

**Training (fitting)** — The learning step, `.fit(...)`, where the model adjusts itself to match the training data. See: [simple_linear_regression](Part%202%20-%20Regression/Section%204%20-%20Simple%20Linear%20Regression/Python/simple_linear_regression.ipynb)

**Training set** — The rows the model learns from. See: [data_preprocessing_template](Part%201%20-%20Data%20Preprocessing/Section%202%20--------------------%20Part%201%20-%20Data%20Preprocessing%20--------------------/Python/data_preprocessing_template.ipynb)

**Underfitting** — The opposite of overfitting: the model is too simple to capture the pattern, so it does badly even on the training data. *Example:* a straight line through the curved salary data (it predicts about $330k for level 6.5). See: [polynomial_regression](Part%202%20-%20Regression/Section%206%20-%20Polynomial%20Regression/Python/polynomial_regression.ipynb)

**Unsupervised learning** — Learning from data with **no** answer column. The computer has to find structure (groups, patterns) on its own. Clustering, association rules and PCA are unsupervised. See: [k_means_clustering](Part%204%20-%20Clustering/Section%2024%20-%20K-Means%20Clustering/Python/k_means_clustering.ipynb)

## Regression

**Coefficient** — The multiplier the model learns for each input, such as the "raise per year" in Salary = start + raise × years. Also called a *weight*. See: [simple_linear_regression](Part%202%20-%20Regression/Section%204%20-%20Simple%20Linear%20Regression/Python/simple_linear_regression.ipynb)

**Decision tree regression** — Predicts a number by asking yes/no questions ("Is the level 6.5 or less?") until it reaches a *leaf*, then gives the average of the training rows in that leaf. Its prediction line looks like a staircase. *Example:* level 6.5 → exactly $150,000. See: [decision_tree_regression](Part%202%20-%20Regression/Section%208%20-%20Decision%20Tree%20Regression/Python/decision_tree_regression.ipynb)

**Degree (polynomial)** — The highest power used in polynomial regression. Degree 4 means level, level², level³ and level⁴. A higher degree gives a bendier curve but risks overfitting. See: [polynomial_regression](Part%202%20-%20Regression/Section%206%20-%20Polynomial%20Regression/Python/polynomial_regression.ipynb)

**Epsilon tube** — In SVR, a "tolerance zone" around the curve. Points inside the tube count as close enough and are ignored; only points outside it push the curve around. `epsilon=0.1` is its half-width in scaled units. See: [support_vector_regression](Part%202%20-%20Regression/Section%207%20-%20Support%20Vector%20Regression%20%28SVR%29/Python/support_vector_regression.ipynb)

**Feature selection (backward elimination)** — Removing inputs that don't help the prediction. It is mentioned in the course but not done in the code. See: [multiple_linear_regression](Part%202%20-%20Regression/Section%205%20-%20Multiple%20Linear%20Regression/Python/multiple_linear_regression.ipynb)

**Intercept** — The starting value of a line: the prediction when every input is 0 (the "starting salary" or "base amount"). See: [simple_linear_regression](Part%202%20-%20Regression/Section%204%20-%20Simple%20Linear%20Regression/Python/simple_linear_regression.ipynb)

**Leaf** — An end point of a decision tree, where the questions stop and a prediction is given. See: [decision_tree_regression](Part%202%20-%20Regression/Section%208%20-%20Decision%20Tree%20Regression/Python/decision_tree_regression.ipynb)

**Least squares** — How linear regression finds the "best" line. For each point, take the vertical gap to the line and square it. Then add them all up. The best line has the smallest total. See: [simple_linear_regression](Part%202%20-%20Regression/Section%204%20-%20Simple%20Linear%20Regression/Python/simple_linear_regression.ipynb)

**Linear regression (simple)** — Predicting a number with a straight line from one input: *y = intercept + coefficient × x*. *Example:* salary from years of experience. See: [simple_linear_regression](Part%202%20-%20Regression/Section%204%20-%20Simple%20Linear%20Regression/Python/simple_linear_regression.ipynb)

**Multiple linear regression** — Linear regression with several inputs, one coefficient per input. *Example:* startup profit from R&D, admin, marketing spend and state. See: [multiple_linear_regression](Part%202%20-%20Regression/Section%205%20-%20Multiple%20Linear%20Regression/Python/multiple_linear_regression.ipynb)

**Polynomial features** — Extra columns made from powers of an input (x, x², x³ …) so that a "linear" model can draw a curve. Built with `PolynomialFeatures`. See: [polynomial_regression](Part%202%20-%20Regression/Section%206%20-%20Polynomial%20Regression/Python/polynomial_regression.ipynb)

**Polynomial regression** — Linear regression on polynomial features, giving a smooth curved line. *Example:* level 6.5 → about $158,862. See: [polynomial_regression](Part%202%20-%20Regression/Section%206%20-%20Polynomial%20Regression/Python/polynomial_regression.ipynb)

**Random forest regression** — Many decision trees, each trained on a random sample of the data, whose predictions are **averaged**. *Example:* level 6.5 → $167,000. See: [random_forest_regression](Part%202%20-%20Regression/Section%209%20-%20Random%20Forest%20Regression/Python/random_forest_regression.ipynb)

**Regression** — Predicting a **number** (a salary, a profit) rather than a category. See: [simple_linear_regression](Part%202%20-%20Regression/Section%204%20-%20Simple%20Linear%20Regression/Python/simple_linear_regression.ipynb)

**Support Vector Regression (SVR)** — Fits a smooth curve with an epsilon tube around it, ignoring small errors inside the tube. Needs feature scaling of both X and y. *Example:* level 6.5 → about $170,370. See: [support_vector_regression](Part%202%20-%20Regression/Section%207%20-%20Support%20Vector%20Regression%20%28SVR%29/Python/support_vector_regression.ipynb)

## Classification

**Bootstrap sample** — A random sample drawn *with replacement*, like pulling names from a hat and putting each one back. Some rows appear twice and some not at all. Each tree in a random forest gets its own bootstrap sample. See: [random_forest_regression](Part%202%20-%20Regression/Section%209%20-%20Random%20Forest%20Regression/Python/random_forest_regression.ipynb)

**C (penalty setting)** — An SVM setting for how strictly mistakes on the training data are punished. A big C tries hard to get every training point right (which risks overfitting). A small C allows a few mistakes in exchange for a smoother, wider boundary. See: [support_vector_machine](Part%203%20-%20Classification/Section%2016%20-%20Support%20Vector%20Machine%20%28SVM%29/Python/support_vector_machine.ipynb)

**Classification** — Predicting a **category** (buy / don't buy, cat / dog, segment 1/2/3) rather than a number. See: [logistic_regression](Part%203%20-%20Classification/Section%2014%20-%20Logistic%20Regression/Python/logistic_regression.ipynb)

**Decision boundary** — The line or curve where the model switches from one answer to the other. On the course charts it is where the red and green areas meet. See: [logistic_regression](Part%203%20-%20Classification/Section%2014%20-%20Logistic%20Regression/Python/logistic_regression.ipynb)

**Decision tree (classification)** — A chain of yes/no questions about the inputs ("Is age over 42?") that ends in a category. Its boundary is made of straight horizontal and vertical edges, like boxes. See: [decision_tree_classification](Part%203%20-%20Classification/Section%2019%20-%20Decision%20Tree%20Classification/Python/decision_tree_classification.ipynb)

**Ensemble** — A team of models whose answers are combined (by voting or averaging) into one. Random forests and XGBoost are ensembles of trees. See: [random_forest_regression](Part%202%20-%20Regression/Section%209%20-%20Random%20Forest%20Regression/Python/random_forest_regression.ipynb)

**Entropy** — A measure of how mixed up a group is. A group of only buyers has entropy 0; a 50/50 mix has the most. A tree using `criterion='entropy'` picks the question that makes the groups as "pure" as possible. See: [decision_tree_classification](Part%203%20-%20Classification/Section%2019%20-%20Decision%20Tree%20Classification/Python/decision_tree_classification.ipynb)

**Euclidean distance** — The straight-line distance between two points, as if measured with a ruler on the chart. K-NN's `metric='minkowski', p=2` is the Euclidean distance. See: [k_nearest_neighbors](Part%203%20-%20Classification/Section%2015%20-%20K-Nearest%20Neighbors%20%28K-NN%29/Python/k_nearest_neighbors.ipynb)

**Gamma** — An RBF-kernel setting for how far each training point's influence reaches. Small gamma gives a smooth boundary; big gamma gives a wiggly one. See: [kernel_svm](Part%203%20-%20Classification/Section%2017%20-%20Kernel%20SVM/Python/kernel_svm.ipynb)

**Gaussian (bell curve)** — A shape where most values sit near the middle and fewer at the edges. `GaussianNB` assumes each input follows a bell curve within each class. See: [naive_bayes](Part%203%20-%20Classification/Section%2018%20-%20Naive%20Bayes/Python/naive_bayes.ipynb)

**K-Nearest Neighbours (K-NN)** — Classifies a new point by a vote of the K most similar training points (here K = 5). There is no formula to learn; it just remembers the training data. See: [k_nearest_neighbors](Part%203%20-%20Classification/Section%2015%20-%20K-Nearest%20Neighbors%20%28K-NN%29/Python/k_nearest_neighbors.ipynb)

**Kernel** — A shortcut that measures how similar two points are, which lets a model draw curved shapes. `'linear'` gives straight lines; `'rbf'` gives smooth curves. See: [support_vector_regression](Part%202%20-%20Regression/Section%207%20-%20Support%20Vector%20Regression%20%28SVR%29/Python/support_vector_regression.ipynb)

**Kernel SVM** — An SVM that uses a non-linear kernel (here RBF), so its boundary can curve around groups. *Example:* 93% accuracy versus 90% for the straight-line SVM. See: [kernel_svm](Part%203%20-%20Classification/Section%2017%20-%20Kernel%20SVM/Python/kernel_svm.ipynb)

**Kernel trick** — Imagine lifting the points into a higher dimension where a flat sheet can separate them. Seen back on the page, the edge of that sheet is a curve. The kernel does this with maths, without ever building the bigger space. See: [kernel_svm](Part%203%20-%20Classification/Section%2017%20-%20Kernel%20SVM/Python/kernel_svm.ipynb)

**Likelihood** — In Naive Bayes, "how typical are this age and salary for buyers?" (and for non-buyers). It is read off each group's bell curves. See: [naive_bayes](Part%203%20-%20Classification/Section%2018%20-%20Naive%20Bayes/Python/naive_bayes.ipynb)

**Logistic regression** — A classifier that turns a weighted sum of the inputs into a probability between 0 and 1 using an S-curve (sigmoid). Above 50% means "yes". Its boundary is a straight line. See: [logistic_regression](Part%203%20-%20Classification/Section%2014%20-%20Logistic%20Regression/Python/logistic_regression.ipynb)

**Margin** — The empty "road" between the SVM's dividing line and the nearest points of each group. An SVM picks the line with the widest possible margin. See: [support_vector_machine](Part%203%20-%20Classification/Section%2016%20-%20Support%20Vector%20Machine%20%28SVM%29/Python/support_vector_machine.ipynb)

**Meshgrid** — A very fine grid of imaginary points covering the whole chart. The model predicts every point, and the colours show its decision regions. Built with `np.meshgrid`. See: [logistic_regression](Part%203%20-%20Classification/Section%2014%20-%20Logistic%20Regression/Python/logistic_regression.ipynb)

**Naive Bayes** — A classifier that weighs up clues using probability: how common each class is (the prior) times how typical the inputs are for that class (the likelihood). It is "naive" because it treats every input as an independent clue. See: [naive_bayes](Part%203%20-%20Classification/Section%2018%20-%20Naive%20Bayes/Python/naive_bayes.ipynb)

**Prior** — In Naive Bayes, how common each class is overall, before looking at the person's details. *Example:* the share of training people who bought. See: [naive_bayes](Part%203%20-%20Classification/Section%2018%20-%20Naive%20Bayes/Python/naive_bayes.ipynb)

**Probability threshold** — The cut-off that turns a probability into a yes/no answer. The course uses 0.5 (above 50% = "yes"). See: [logistic_regression](Part%203%20-%20Classification/Section%2014%20-%20Logistic%20Regression/Python/logistic_regression.ipynb)

**Random forest (classification)** — Many decision trees, each trained on a random slice of the data, that **vote**; the most common answer wins. See: [random_forest_classification](Part%203%20-%20Classification/Section%2020%20-%20Random%20Forest%20Classification/Python/random_forest_classification.ipynb)

**RBF kernel** — Radial Basis Function (Gaussian) kernel. It scores two points close to 1 when they are near each other and close to 0 when they are far apart. Each point spreads a "bubble of influence", which lets boundaries curve. See: [support_vector_regression](Part%202%20-%20Regression/Section%207%20-%20Support%20Vector%20Regression%20%28SVR%29/Python/support_vector_regression.ipynb)

**Support vector** — The training points closest to the edge of the SVM's margin. They alone decide where the line goes. See: [support_vector_machine](Part%203%20-%20Classification/Section%2016%20-%20Support%20Vector%20Machine%20%28SVM%29/Python/support_vector_machine.ipynb)

**Support Vector Machine (SVM)** — A classifier that draws the dividing line with the widest possible gap (margin) between the classes. With `kernel='linear'` the line is straight. See: [support_vector_machine](Part%203%20-%20Classification/Section%2016%20-%20Support%20Vector%20Machine%20%28SVM%29/Python/support_vector_machine.ipynb)

## Measuring results

**Accuracy** — Correct guesses ÷ all guesses. *Example:* (65 + 24) ÷ 100 = 0.89 for logistic regression. It can mislead when one class is much more common than the other. See: [logistic_regression](Part%203%20-%20Classification/Section%2014%20-%20Logistic%20Regression/Python/logistic_regression.ipynb)

**Confusion matrix** — A table counting right and wrong guesses. Rows are the real answer and columns are the predicted answer; the diagonal holds the correct ones. *Example:* `[[65 3] [8 24]]`. See: [logistic_regression](Part%203%20-%20Classification/Section%2014%20-%20Logistic%20Regression/Python/logistic_regression.ipynb)

**False negative** — The real answer was "yes" but the model said "no" (a missed buyer, a missed leaver, a missed cancer). Bottom-left of a 2×2 confusion matrix. See: [logistic_regression](Part%203%20-%20Classification/Section%2014%20-%20Logistic%20Regression/Python/logistic_regression.ipynb)

**False positive** — The real answer was "no" but the model said "yes" (a false alarm). Top-right of a 2×2 confusion matrix. See: [logistic_regression](Part%203%20-%20Classification/Section%2014%20-%20Logistic%20Regression/Python/logistic_regression.ipynb)

**Standard deviation** — How spread out a set of numbers is. In k-fold cross validation, a small standard deviation means the model scores consistently across the folds. *Example:* 6.57%. See: [k_fold_cross_validation](Part%2010%20-%20Model%20Selection%20_%20Boosting/Section%2048%20-%20Model%20Selection/Python/k_fold_cross_validation.ipynb)

**True negative** — The real answer was "no" and the model said "no". Top-left of a 2×2 confusion matrix. See: [logistic_regression](Part%203%20-%20Classification/Section%2014%20-%20Logistic%20Regression/Python/logistic_regression.ipynb)

**True positive** — The real answer was "yes" and the model said "yes". Bottom-right of a 2×2 confusion matrix. See: [logistic_regression](Part%203%20-%20Classification/Section%2014%20-%20Logistic%20Regression/Python/logistic_regression.ipynb)

## Clustering

**Agglomerative clustering** — The bottom-up kind of hierarchical clustering: start with every point as its own group and keep merging the two closest groups. See: [hierarchical_clustering](Part%204%20-%20Clustering/Section%2025%20-%20Hierarchical%20Clustering/Python/hierarchical_clustering.ipynb)

**Centroid** — The centre point of a cluster (the "flag" everyone walks to in K-Means). Drawn as big yellow dots in the K-Means chart. See: [k_means_clustering](Part%204%20-%20Clustering/Section%2024%20-%20K-Means%20Clustering/Python/k_means_clustering.ipynb)

**Cluster** — A group of similar points that the algorithm found on its own. The cluster numbers are arbitrary names, not right answers. See: [k_means_clustering](Part%204%20-%20Clustering/Section%2024%20-%20K-Means%20Clustering/Python/k_means_clustering.ipynb)

**Clustering** — Splitting data into groups of similar items without being told what the groups are. *Example:* 200 mall shoppers grouped by income and spending score. See: [k_means_clustering](Part%204%20-%20Clustering/Section%2024%20-%20K-Means%20Clustering/Python/k_means_clustering.ipynb)

**Dendrogram** — A tree diagram that shows every merge in hierarchical clustering. The height of each join shows how far apart the two groups were. Cutting across the longest uncrossed vertical lines suggests how many clusters to use (5 here). See: [hierarchical_clustering](Part%204%20-%20Clustering/Section%2025%20-%20Hierarchical%20Clustering/Python/hierarchical_clustering.ipynb)

**Elbow method** — A way to choose K for K-Means. Plot WCSS for K = 1 to 10 and pick the point where the line bends from steep to flat (the "elbow"). *Example:* the elbow is at 5. See: [k_means_clustering](Part%204%20-%20Clustering/Section%2024%20-%20K-Means%20Clustering/Python/k_means_clustering.ipynb)

**Hierarchical clustering** — Builds a family tree of groups by repeatedly merging the closest ones, then cuts the tree to get the number of clusters you want. See: [hierarchical_clustering](Part%204%20-%20Clustering/Section%2025%20-%20Hierarchical%20Clustering/Python/hierarchical_clustering.ipynb)

**K-Means** — Place K centroids, assign every point to its nearest centroid, move each centroid to the middle of its points, and repeat until nothing changes. See: [k_means_clustering](Part%204%20-%20Clustering/Section%2024%20-%20K-Means%20Clustering/Python/k_means_clustering.ipynb)

**K-means++** — A smart way to place the starting centroids spread out from each other, which avoids bad groupings caused by unlucky starting spots (`init='k-means++'`). See: [k_means_clustering](Part%204%20-%20Clustering/Section%2024%20-%20K-Means%20Clustering/Python/k_means_clustering.ipynb)

**Ward linkage** — A merging rule for hierarchical clustering: merge the two groups whose union is the least spread out. It gives compact, evenly sized groups, much like K-Means. See: [hierarchical_clustering](Part%204%20-%20Clustering/Section%2025%20-%20Hierarchical%20Clustering/Python/hierarchical_clustering.ipynb)

**WCSS (within-cluster sum of squares)** — Add up the squared distance from each point to its own centroid. Smaller means tighter groups. scikit-learn calls it `inertia_`. See: [k_means_clustering](Part%204%20-%20Clustering/Section%2024%20-%20K-Means%20Clustering/Python/k_means_clustering.ipynb)

## Association rules

**Apriori** — An algorithm that finds "people who buy A also buy B" rules, scored by support, confidence and lift. It saves time by throwing away rare combinations early. See: [apriori](Part%205%20-%20Association%20Rule%20Learning/Section%2028%20-%20Apriori/Python/apriori.ipynb)

**Association rule learning** — Finding items that tend to occur together, such as products in the same shopping basket. See: [apriori](Part%205%20-%20Association%20Rule%20Learning/Section%2028%20-%20Apriori/Python/apriori.ipynb)

**Confidence** — Out of the baskets that contain A, the share that also contain B. *Example:* 29% of light-cream buyers also bought chicken (confidence 0.29). See: [apriori](Part%205%20-%20Association%20Rule%20Learning/Section%2028%20-%20Apriori/Python/apriori.ipynb)

**Eclat** — A simpler cousin of Apriori that ranks sets of items by **support only**, with no direction ("A and B go together"). See: [eclat](Part%205%20-%20Association%20Rule%20Learning/Section%2029%20-%20Eclat/Python/eclat.ipynb)

**Left-hand side / right-hand side (LHS / RHS)** — The two halves of a rule: "if a basket has the LHS item, it probably also has the RHS item". See: [apriori](Part%205%20-%20Association%20Rule%20Learning/Section%2028%20-%20Apriori/Python/apriori.ipynb)

**Lift** — How much more likely B is when A is in the basket, compared with normal. Lift 1 means no link; lift 3 means three times more likely. *Example:* fromage blanc → honey has lift 5.16. See: [apriori](Part%205%20-%20Association%20Rule%20Learning/Section%2028%20-%20Apriori/Python/apriori.ipynb)

**Support** — The share of all baskets that contain the item (or items). *Example:* herb & pepper + ground beef appear together in about 1.6% of the 7,501 baskets (support 0.016). See: [apriori](Part%205%20-%20Association%20Rule%20Learning/Section%2028%20-%20Apriori/Python/apriori.ipynb)

**Transaction (basket)** — One customer's list of purchased items; one row of the shopping data. See: [apriori](Part%205%20-%20Association%20Rule%20Learning/Section%2028%20-%20Apriori/Python/apriori.ipynb)

## Reinforcement learning

**Beta distribution** — A curve describing where an ad's true click rate probably lies, based on its wins and losses so far. Thompson Sampling draws a random guess from it (`random.betavariate(wins + 1, losses + 1)`). See: [thompson_sampling](Part%206%20-%20Reinforcement%20Learning/Section%2033%20-%20Thompson%20Sampling/Python/thompson_sampling.ipynb)

**Click-through rate (CTR)** — The share of people who click an ad when it is shown. In the data, Ad 5 has the highest, about 27%. See: [upper_confidence_bound](Part%206%20-%20Reinforcement%20Learning/Section%2032%20-%20Upper%20Confidence%20Bound%20%28UCB%29/Python/upper_confidence_bound.ipynb)

**Deterministic** — Always gives the same result for the same data. UCB is deterministic; Thompson Sampling is random, so its results change a little each run. See: [thompson_sampling](Part%206%20-%20Reinforcement%20Learning/Section%2033%20-%20Thompson%20Sampling/Python/thompson_sampling.ipynb)

**Exploration vs exploitation** — The trade-off between *trying* options you know little about (exploring) and *using* the one that looks best so far (exploiting). See: [upper_confidence_bound](Part%206%20-%20Reinforcement%20Learning/Section%2032%20-%20Upper%20Confidence%20Bound%20%28UCB%29/Python/upper_confidence_bound.ipynb)

**Multi-armed bandit** — The "10 slot machines" problem: pick one option per round, see only its reward, and try to win as much as possible overall. *Example:* choosing which of 10 ads to show each visitor. See: [upper_confidence_bound](Part%206%20-%20Reinforcement%20Learning/Section%2032%20-%20Upper%20Confidence%20Bound%20%28UCB%29/Python/upper_confidence_bound.ipynb)

**Reinforcement learning** — Learning by acting: the program makes a choice, gets a reward (or not), and improves its next choice. There is no ready-made training table. See: [upper_confidence_bound](Part%206%20-%20Reinforcement%20Learning/Section%2032%20-%20Upper%20Confidence%20Bound%20%28UCB%29/Python/upper_confidence_bound.ipynb)

**Reward** — The feedback after an action, here 1 for a click and 0 for no click. See: [upper_confidence_bound](Part%206%20-%20Reinforcement%20Learning/Section%2032%20-%20Upper%20Confidence%20Bound%20%28UCB%29/Python/upper_confidence_bound.ipynb)

**Round** — One step of a bandit run. Here each round is one visitor (10,000 rounds). See: [upper_confidence_bound](Part%206%20-%20Reinforcement%20Learning/Section%2032%20-%20Upper%20Confidence%20Bound%20%28UCB%29/Python/upper_confidence_bound.ipynb)

**Thompson Sampling** — For each ad, draw a random guess of its click rate from its Beta distribution and show the ad with the highest guess. Ads you know little about give wild guesses, so they still get tried now and then. See: [thompson_sampling](Part%206%20-%20Reinforcement%20Learning/Section%2033%20-%20Thompson%20Sampling/Python/thompson_sampling.ipynb)

**Upper Confidence Bound (UCB)** — Score each ad as "average reward so far + an uncertainty bonus" and always show the highest score. The bonus shrinks as an ad is shown more often. See: [upper_confidence_bound](Part%206%20-%20Reinforcement%20Learning/Section%2032%20-%20Upper%20Confidence%20Bound%20%28UCB%29/Python/upper_confidence_bound.ipynb)

## Text (NLP)

**Bag of words** — Turning each text into a row of word counts, one column per word, and ignoring word order (as if the words were tipped into a bag). See: [natural_language_processing](Part%207%20-%20Natural%20Language%20Processing/Section%2036%20-%20Natural%20Language%20Processing/Python/natural_language_processing.ipynb)

**Corpus** — The full collection of (cleaned) texts. In the code it is the list `corpus`. See: [natural_language_processing](Part%207%20-%20Natural%20Language%20Processing/Section%2036%20-%20Natural%20Language%20Processing/Python/natural_language_processing.ipynb)

**CountVectorizer** — The scikit-learn tool that builds the bag-of-words table. `max_features=1500` keeps only the 1,500 most frequent words. See: [natural_language_processing](Part%207%20-%20Natural%20Language%20Processing/Section%2036%20-%20Natural%20Language%20Processing/Python/natural_language_processing.ipynb)

**Natural Language Processing (NLP)** — Getting computers to work with human language, here to tell positive restaurant reviews from negative ones. See: [natural_language_processing](Part%207%20-%20Natural%20Language%20Processing/Section%2036%20-%20Natural%20Language%20Processing/Python/natural_language_processing.ipynb)

**Regular expression** — A pattern for matching text. `'[^a-zA-Z]'` means "anything that is not a letter", which the code replaces with a space. See: [natural_language_processing](Part%207%20-%20Natural%20Language%20Processing/Section%2036%20-%20Natural%20Language%20Processing/Python/natural_language_processing.ipynb)

**Sparse table** — A table where most cells are 0. A bag of words is sparse because each short review uses only a handful of the 1,500 words. See: [natural_language_processing](Part%207%20-%20Natural%20Language%20Processing/Section%2036%20-%20Natural%20Language%20Processing/Python/natural_language_processing.ipynb)

**Stemming** — Cutting words down to their root so different forms count as one word ("loved", "loving" → "love"). Stems aren't always real words ("tasty" → "tasti"). See: [natural_language_processing](Part%207%20-%20Natural%20Language%20Processing/Section%2036%20-%20Natural%20Language%20Processing/Python/natural_language_processing.ipynb)

**Stopwords** — Very common words ("the", "is", "this") that carry little meaning and are removed. The course keeps **"not"**, because it flips the meaning ("not good"). See: [natural_language_processing](Part%207%20-%20Natural%20Language%20Processing/Section%2036%20-%20Natural%20Language%20Processing/Python/natural_language_processing.ipynb)

**Tokenisation** — Splitting text into separate words (tokens). The course does this with `.split()`. See: [natural_language_processing](Part%207%20-%20Natural%20Language%20Processing/Section%2036%20-%20Natural%20Language%20Processing/Python/natural_language_processing.ipynb)

## Neural networks

**Activation function** — The rule a neuron applies to its total before passing it on. It lets the network learn curved, non-straight patterns. See: [artificial_neural_network](Part%208%20-%20Deep%20Learning/Section%2039%20-%20Artificial%20Neural%20Networks%20%28ANN%29/Python/artificial_neural_network.ipynb)

**Adam** — A popular optimiser: a smart version of gradient descent that adjusts its own step size as it learns. See: [artificial_neural_network](Part%208%20-%20Deep%20Learning/Section%2039%20-%20Artificial%20Neural%20Networks%20%28ANN%29/Python/artificial_neural_network.ipynb)

**Artificial Neural Network (ANN)** — Layers of simple calculators (neurons) linked by weights. The weights are tuned during training until the outputs come out right. *Example:* 12 inputs → 6 → 6 → 1. See: [artificial_neural_network](Part%208%20-%20Deep%20Learning/Section%2039%20-%20Artificial%20Neural%20Networks%20%28ANN%29/Python/artificial_neural_network.ipynb)

**Batch size** — How many examples the network looks at before each weight update. *Example:* 32, so 8,000 customers make 250 batches per epoch. See: [artificial_neural_network](Part%208%20-%20Deep%20Learning/Section%2039%20-%20Artificial%20Neural%20Networks%20%28ANN%29/Python/artificial_neural_network.ipynb)

**Binary cross-entropy** — The loss function for yes/no problems. It punishes confident wrong answers heavily. See: [artificial_neural_network](Part%208%20-%20Deep%20Learning/Section%2039%20-%20Artificial%20Neural%20Networks%20%28ANN%29/Python/artificial_neural_network.ipynb)

**Channels (RGB)** — The three colour numbers (red, green, blue) for each pixel, so a 64 × 64 colour image is 64 × 64 × 3 numbers. See: [convolutional_neural_network](Part%208%20-%20Deep%20Learning/Section%2040%20-%20Convolutional%20Neural%20Networks%20%28CNN%29/Python/convolutional_neural_network.ipynb)

**Convolution** — Sliding a small window (filter) across an image and checking how well each spot matches a pattern. See: [convolutional_neural_network](Part%208%20-%20Deep%20Learning/Section%2040%20-%20Convolutional%20Neural%20Networks%20%28CNN%29/Python/convolutional_neural_network.ipynb)

**Convolutional Neural Network (CNN)** — A neural network built for images. Convolution and pooling layers find visual features, then ordinary dense layers make the decision. See: [convolutional_neural_network](Part%208%20-%20Deep%20Learning/Section%2040%20-%20Convolutional%20Neural%20Networks%20%28CNN%29/Python/convolutional_neural_network.ipynb)

**Deep learning** — Machine learning with neural networks that have two or more hidden layers. See: [artificial_neural_network](Part%208%20-%20Deep%20Learning/Section%2039%20-%20Artificial%20Neural%20Networks%20%28ANN%29/Python/artificial_neural_network.ipynb)

**Dense (fully connected) layer** — A layer where every neuron receives every number from the layer before. See: [artificial_neural_network](Part%208%20-%20Deep%20Learning/Section%2039%20-%20Artificial%20Neural%20Networks%20%28ANN%29/Python/artificial_neural_network.ipynb)

**Epoch** — One full pass through the whole training set. *Example:* 100 epochs for the ANN, 25 for the CNN. See: [artificial_neural_network](Part%208%20-%20Deep%20Learning/Section%2039%20-%20Artificial%20Neural%20Networks%20%28ANN%29/Python/artificial_neural_network.ipynb)

**Feature map** — The output of one filter: a small image showing *where* its pattern appeared (62 × 62 after the first convolution). See: [convolutional_neural_network](Part%208%20-%20Deep%20Learning/Section%2040%20-%20Convolutional%20Neural%20Networks%20%28CNN%29/Python/convolutional_neural_network.ipynb)

**Filter (feature detector)** — A small grid of weights (here 3 × 3) that looks for one pattern, such as an edge or a curve. The network learns what each filter looks for. See: [convolutional_neural_network](Part%208%20-%20Deep%20Learning/Section%2040%20-%20Convolutional%20Neural%20Networks%20%28CNN%29/Python/convolutional_neural_network.ipynb)

**Flattening** — Laying a stack of feature maps out as one long list of numbers (14 × 14 × 32 = 6,272) so dense layers can use it. See: [convolutional_neural_network](Part%208%20-%20Deep%20Learning/Section%2040%20-%20Convolutional%20Neural%20Networks%20%28CNN%29/Python/convolutional_neural_network.ipynb)

**Gradient descent** — Learning by small steps: work out which direction makes the error smaller and nudge the weights that way. See: [artificial_neural_network](Part%208%20-%20Deep%20Learning/Section%2039%20-%20Artificial%20Neural%20Networks%20%28ANN%29/Python/artificial_neural_network.ipynb)

**Hidden layer** — Any layer between the inputs and the output. See: [artificial_neural_network](Part%208%20-%20Deep%20Learning/Section%2039%20-%20Artificial%20Neural%20Networks%20%28ANN%29/Python/artificial_neural_network.ipynb)

**Image augmentation** — Making small random changes (slant, zoom, flip) to training images each time they are used, so the network can't just memorise the exact photos. See: [convolutional_neural_network](Part%208%20-%20Deep%20Learning/Section%2040%20-%20Convolutional%20Neural%20Networks%20%28CNN%29/Python/convolutional_neural_network.ipynb)

**Layer** — A row of neurons that all work on the same inputs and pass their outputs to the next layer. See: [artificial_neural_network](Part%208%20-%20Deep%20Learning/Section%2039%20-%20Artificial%20Neural%20Networks%20%28ANN%29/Python/artificial_neural_network.ipynb)

**Loss** — A number measuring how wrong the network's guesses are. Training tries to make it smaller. *Example:* the ANN's loss falls from about 0.80 to about 0.33. See: [artificial_neural_network](Part%208%20-%20Deep%20Learning/Section%2039%20-%20Artificial%20Neural%20Networks%20%28ANN%29/Python/artificial_neural_network.ipynb)

**Max pooling** — Shrinking a feature map by keeping only the largest value in each small square (2 × 2). This halves its size and makes the network care less about exact positions. See: [convolutional_neural_network](Part%208%20-%20Deep%20Learning/Section%2040%20-%20Convolutional%20Neural%20Networks%20%28CNN%29/Python/convolutional_neural_network.ipynb)

**Neuron** — A tiny calculator: multiply each input by a weight, add them up, apply an activation function and pass the result on. See: [artificial_neural_network](Part%208%20-%20Deep%20Learning/Section%2039%20-%20Artificial%20Neural%20Networks%20%28ANN%29/Python/artificial_neural_network.ipynb)

**Optimiser** — The method used to update the weights during training (the course uses Adam). See: [artificial_neural_network](Part%208%20-%20Deep%20Learning/Section%2039%20-%20Artificial%20Neural%20Networks%20%28ANN%29/Python/artificial_neural_network.ipynb)

**ReLU (rectifier)** — An activation function: if the total is negative output 0, otherwise pass it on unchanged. It is used in hidden layers. See: [artificial_neural_network](Part%208%20-%20Deep%20Learning/Section%2039%20-%20Artificial%20Neural%20Networks%20%28ANN%29/Python/artificial_neural_network.ipynb)

**Sigmoid** — An S-shaped function that squashes any number into the range 0 to 1, so the result can be read as a probability. It is used in logistic regression and in the output layer for yes/no answers. See: [logistic_regression](Part%203%20-%20Classification/Section%2014%20-%20Logistic%20Regression/Python/logistic_regression.ipynb)

**Softmax** — The output activation used instead of sigmoid when there are three or more possible answers. It is mentioned but not used in the course. See: [artificial_neural_network](Part%208%20-%20Deep%20Learning/Section%2039%20-%20Artificial%20Neural%20Networks%20%28ANN%29/Python/artificial_neural_network.ipynb)

**Stride** — How many pixels a window moves each step. `strides=2` in pooling means the 2 × 2 squares don't overlap. See: [convolutional_neural_network](Part%208%20-%20Deep%20Learning/Section%2040%20-%20Convolutional%20Neural%20Networks%20%28CNN%29/Python/convolutional_neural_network.ipynb)

**Validation data** — Data used to check the network after each epoch but never used for learning (`validation_data = test_set`). See: [convolutional_neural_network](Part%208%20-%20Deep%20Learning/Section%2040%20-%20Convolutional%20Neural%20Networks%20%28CNN%29/Python/convolutional_neural_network.ipynb)

**Weight** — A number on each connection that says how much an input matters. Training is mostly the job of finding good weights. See: [artificial_neural_network](Part%208%20-%20Deep%20Learning/Section%2039%20-%20Artificial%20Neural%20Networks%20%28ANN%29/Python/artificial_neural_network.ipynb)

## Dimensionality reduction

**Dimension** — Another word for an input column. The wine data has 13 dimensions; reducing it to 2 lets us draw it on a flat chart. See: [principal_component_analysis](Part%209%20-%20Dimensionality%20Reduction/Section%2043%20-%20Principal%20Component%20Analysis%20%28PCA%29/Python/principal_component_analysis.ipynb)

**Dimensionality reduction** — Squeezing many input columns into a few new ones while keeping as much useful information as possible. See: [principal_component_analysis](Part%209%20-%20Dimensionality%20Reduction/Section%2043%20-%20Principal%20Component%20Analysis%20%28PCA%29/Python/principal_component_analysis.ipynb)

**Explained variance** — The share of the data's total spread that a principal component captures. PC1 explains the most, PC2 the next most. scikit-learn stores it in `explained_variance_ratio_` (the notebooks don't print it). See: [principal_component_analysis](Part%209%20-%20Dimensionality%20Reduction/Section%2043%20-%20Principal%20Component%20Analysis%20%28PCA%29/Python/principal_component_analysis.ipynb)

**Kernel PCA** — PCA combined with the kernel trick (here RBF), so it can follow curved patterns. Like PCA, it does not use the labels. See: [kernel_pca](Part%209%20-%20Dimensionality%20Reduction/Section%2045%20-%20Kernel%20PCA/Python/kernel_pca.ipynb)

**Linear Discriminant Analysis (LDA)** — Finds the directions that best **separate the known classes**. It uses the labels, so it is supervised. With 3 classes it can make at most 3 − 1 = 2 directions. See: [linear_discriminant_analysis](Part%209%20-%20Dimensionality%20Reduction/Section%2044%20-%20Linear%20Discriminant%20Analysis%20%28LDA%29/Python/linear_discriminant_analysis.ipynb)

**Linear discriminant (LD1, LD2)** — One of the new columns LDA creates: a weighted mix of all the original columns. See: [linear_discriminant_analysis](Part%209%20-%20Dimensionality%20Reduction/Section%2044%20-%20Linear%20Discriminant%20Analysis%20%28LDA%29/Python/linear_discriminant_analysis.ipynb)

**Principal component (PC1, PC2)** — One of the new columns PCA creates: a weighted mix of all the original columns. PC1 points the way the data varies most; PC2 is the next-biggest direction that doesn't repeat PC1. See: [principal_component_analysis](Part%209%20-%20Dimensionality%20Reduction/Section%2043%20-%20Principal%20Component%20Analysis%20%28PCA%29/Python/principal_component_analysis.ipynb)

**Principal Component Analysis (PCA)** — Finds the directions in which the data is most spread out and keeps the top few. It does **not** use the labels, so it is unsupervised. Scale the data first. See: [principal_component_analysis](Part%209%20-%20Dimensionality%20Reduction/Section%2043%20-%20Principal%20Component%20Analysis%20%28PCA%29/Python/principal_component_analysis.ipynb)

## Tuning & boosting

**Fold** — One of the k equal pieces the training data is cut into for cross validation. See: [k_fold_cross_validation](Part%2010%20-%20Model%20Selection%20_%20Boosting/Section%2048%20-%20Model%20Selection/Python/k_fold_cross_validation.ipynb)

**Gradient boosting** — Building many small trees one after another, each one trained to fix the mistakes left by the trees before it. Added together, many weak guessers become one strong one. See: [xg_boost](Part%2010%20-%20Model%20Selection%20_%20Boosting/Section%2049%20-%20XGBoost/Python/xg_boost.ipynb)

**Grid search** — Trying every combination of hyperparameter values from a list and keeping the best, judged by cross validation. *Example:* 40 combinations → best C = 0.5, gamma = 0.6, RBF kernel, 90.67%. See: [grid_search](Part%2010%20-%20Model%20Selection%20_%20Boosting/Section%2048%20-%20Model%20Selection/Python/grid_search.ipynb)

**Hyperparameter** — A setting you choose *before* training (like C, gamma or the kernel), as opposed to the weights the model learns itself. See: [grid_search](Part%2010%20-%20Model%20Selection%20_%20Boosting/Section%2048%20-%20Model%20Selection/Python/grid_search.ipynb)

**k-Fold cross validation** — Cut the training data into k pieces (here 10). Train k times, each time testing on a different piece, then average the scores. This is fairer than one test. *Example:* 90.33% average, 6.57% standard deviation. See: [k_fold_cross_validation](Part%2010%20-%20Model%20Selection%20_%20Boosting/Section%2048%20-%20Model%20Selection/Python/k_fold_cross_validation.ipynb)

**n_jobs = -1** — Tells scikit-learn to use all your computer's processor cores, so the many training runs in grid search finish faster. See: [grid_search](Part%2010%20-%20Model%20Selection%20_%20Boosting/Section%2048%20-%20Model%20Selection/Python/grid_search.ipynb)

**XGBoost** — "eXtreme Gradient Boosting", a fast and popular gradient-boosting library that often wins competitions on table-shaped data. It needs no feature scaling. See: [xg_boost](Part%2010%20-%20Model%20Selection%20_%20Boosting/Section%2049%20-%20XGBoost/Python/xg_boost.ipynb)

## Tools & libraries

**apyori** — A small library that runs the Apriori algorithm. Installed with `pip install apyori`. See: [apriori](Part%205%20-%20Association%20Rule%20Learning/Section%2028%20-%20Apriori/Python/apriori.ipynb)

**Keras** — The easy building-block part of TensorFlow (`tf.keras`) for making neural networks. See: [artificial_neural_network](Part%208%20-%20Deep%20Learning/Section%2039%20-%20Artificial%20Neural%20Networks%20%28ANN%29/Python/artificial_neural_network.ipynb)

**matplotlib** — Draws the charts. See: [data_preprocessing_tools](Part%201%20-%20Data%20Preprocessing/Section%202%20--------------------%20Part%201%20-%20Data%20Preprocessing%20--------------------/Python/data_preprocessing_tools.ipynb)

**NLTK** — Natural Language Toolkit. The course uses its stopword list and Porter stemmer. See: [natural_language_processing](Part%207%20-%20Natural%20Language%20Processing/Section%2036%20-%20Natural%20Language%20Processing/Python/natural_language_processing.ipynb)

**numpy** — Fast maths on lists and grids of numbers. It also provides `np.nan` for missing values. See: [data_preprocessing_tools](Part%201%20-%20Data%20Preprocessing/Section%202%20--------------------%20Part%201%20-%20Data%20Preprocessing%20--------------------/Python/data_preprocessing_tools.ipynb)

**pandas** — Reads and handles tables, like Excel in Python (`pd.read_csv`). See: [data_preprocessing_tools](Part%201%20-%20Data%20Preprocessing/Section%202%20--------------------%20Part%201%20-%20Data%20Preprocessing%20--------------------/Python/data_preprocessing_tools.ipynb)

**scikit-learn (sklearn)** — The main machine-learning library in this course: models, scalers, encoders, train/test split and scores. See: [data_preprocessing_tools](Part%201%20-%20Data%20Preprocessing/Section%202%20--------------------%20Part%201%20-%20Data%20Preprocessing%20--------------------/Python/data_preprocessing_tools.ipynb)

**TensorFlow** — Google's deep-learning library, used for the ANN and CNN. See: [artificial_neural_network](Part%208%20-%20Deep%20Learning/Section%2039%20-%20Artificial%20Neural%20Networks%20%28ANN%29/Python/artificial_neural_network.ipynb)

---

New to the course? Start with the [Learning Path](LEARNING_PATH.md).
