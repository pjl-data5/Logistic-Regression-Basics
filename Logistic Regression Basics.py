
# =============================================================================
#  Basics: Machine learning optimisation
# =============================================================================
# =============================================================================
# Consider this basic example from an online dataset with 
# a binary outcome variable. We are interested on how certain 
# variables affects this outcome variable GRE (Graduate Record Exam scores), 
# GPA (grade point average) and prestige of an academic institution,
# effect admission into graduate school. The outcome variable, 
# admit/don’t admit, is binary.
# =============================================================================


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
#import the data
df = pd.read_csv('https://stats.idre.ucla.edu/stat/data/binary.csv')
df.info()

# Rename the 'rank' column to 'prestige'
df.columns = ["admit", "gre", "gpa", "prestige"]    

#==============
# EDA
#==============
# Simple stats
df.describe()

# Make a crosstable of prestige and admit variables
cross_tab_prestige = pd.crosstab(df['admit'], df['prestige'], rownames=['admit'])
cross_tab_prestige


# The prestige variable should be treated as a categorical varibale. 
# These variables needs to first be transformed into dummy variables.
# Dummify
dummy_prestige = pd.get_dummies(df['prestige'], prefix='prestige')
dummy_prestige

# Make workable df
print(df['prestige'].unique())
# If there are three dummy variables for 4 categories, one may be dropped as it is redundant 
# drop 'prestige_4'
cols_to_keep = ['admit', 'gre', 'gpa']
data = df[cols_to_keep].join(dummy_prestige.loc[:, :'prestige_3']) 
data

# Use the data frame to make it workable for regression. From the problem description above denote: 
#     - X -> independent variables (data features)
#     - y -> dependent variable (outcome)

# Create variables
train_cols = data.columns[1:]
X = data[train_cols]
y = data['admit']

# Split into an 80% training and 20% test 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#====================
# Logistic Regression
#====================

# Create an instance of of the Logistic regression model
# Visit documentation for the choice of solver.
# Since we have a few independant variable use lbfgs
logistic = LogisticRegression(solver='lbfgs')
# Fit the model to the training data 
logistic_model = logistic.fit(X_train, y_train)
# Accuracy on the training data
acc = logistic_model.score(X_train, y_train)
# Ratio of training examples in class 1 
ratio_class1 = y_train.mean()
print(acc)
# This indicates successful admittance
print(ratio_class1)
#====================
# Prediction
#====================

# Predicted class labels 
predicted = logistic_model.predict(X_test)
# Predicted class probabilities
probs = logistic_model.predict_proba(X_test)
# Sccuracy score on the test set
acc_score = metrics.accuracy_score(y_test, predicted)
# Area under the curve
auc_score = metrics.roc_auc_score(y_test, probs[:, 1])
print(metrics.classification_report(y_test, predicted))