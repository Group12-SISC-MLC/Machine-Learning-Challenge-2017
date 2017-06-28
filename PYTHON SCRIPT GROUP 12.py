from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np

# This is the iris object with the loaded iris data (
iris = load_iris_Iris-setosa.csv

# Feature variables are in this dataframe (df)
df = pd.DataFrame(iris.data, columns=iris.feature_names)

# View the 150 rows above
df.head()

# We are trying to predict the iris setosa 
df['setosa'] = pd.Categorical.from_codes(iris.target, iris.target_names)

# View the 150 rows above
df.head()

# To determine if values indicate a true or false
if y = 1, True
    else, False

# View the 150 rows above
df.head()

# These are two data frames, one with training rows and one with test rows
train, test = df[df['is_train']==True], df[df['is_train']==False]

# These are the number of observations for the test and training dataframes
print('Number of observations in the training data:', len(train))
print('Number of observations in the test data:',len(test))
Number of observations in the training data: 98
Number of observations in the test data: 52
# Total observations = 150 (from data file)

# List of the feature column's names
number of features = df.columns[:5]

features
Index(['X1, X2, X3, X4, Y1'],
      dtype='object')
# train['iris.setosa'] contains the actual values. 


# Create a random forest classifier. By convention, clf means 'classifier'
clf = RandomForestClassifier(n_values=5)

# Train the classifer to relate it to Y
clf.fit(train[features], y)


# The predicted probabilities of the first 10 observations
clf.predict_proba(test[features])[0:10]

# View the ACTUAL values for the first five observations
test['values'].head()

# Create confusion matrix
pd.crosstab(test['values'], preds, rownames=['Actual values'], colnames=['Predicted values'])

# View a list of the features and their importance scores (assessment of test results)
list(zip(train[features], clf.feature_importances_))

