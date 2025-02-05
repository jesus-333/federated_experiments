"""
Non-federated version of XGBoost
https://www.datacamp.com/tutorial/xgboost-in-python

In this tutorial, we will first try to predict diamond prices using their physical measurements
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Imports

import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Settings

random_seed = None
num_boost_round = 1000
early_stopping_rounds = 50
n_fold = 5

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Get dataset 

diamonds = sns.load_dataset("diamonds")

print("First rows of dataset")
print(diamonds.head())

print("\nDataset Size = {}\n".format(diamonds.shape))

print("Dataset stats:")
print(diamonds.describe())

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Preprocess

# Extract feature and target arrays
X, y = diamonds.drop('price', axis = 1), diamonds[['price']]

# Extract text features
cats = X.select_dtypes(exclude = np.number).columns.tolist()

# Convert to Pandas category (https://pandas.pydata.org/docs/user_guide/categorical.html)
for col in cats: X[col] = X[col].astype('category')

# Split the data
# Note that the function train_test_split can have a parameter called test_size.
# If you not specify it by default will be set to 0.25 ---> So by defualt the size of the test set is 25% of the original dataset size
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = random_seed)


# Create regression matrices (xgboost data type)
# More details here : https://stackoverflow.com/questions/70127049/what-is-the-use-of-dmatrix
dtrain_reg = xgb.DMatrix(X_train, y_train, enable_categorical = True)
dtest_reg = xgb.DMatrix(X_test, y_test, enable_categorical = True)

# Define hyperparameters
params = {"objective": "reg:squarederror", "tree_method": "hist"}

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Training 1 : Basic

model = xgb.train(
    params = params,
    dtrain = dtrain_reg,
    num_boost_round = num_boost_round
)

preds = model.predict(dtest_reg)
rmse = mean_squared_error(y_test, preds) 
print(f"\nRMSE of the base model: {rmse:.3f}\n")

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Training 2 : Incorporate validation into the model train 
# N.b. This does not change final performance

print("\n- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - ")
print("Validation incorporated in model's train")

# Array with train data (position 0) and validaiton data (position 1)
# If passed to the train function this data will be used for the evaluation.
# Basically at each round the loss will be computed for this round and printed with the assigend label
# The reason to put the train data inside is to have a comparison of the loss on train data vs the loss on validation data for each round
evals = [(dtrain_reg, "train"), (dtest_reg, "validation")]

model = xgb.train(
    params = params,
    dtrain = dtrain_reg,
    num_boost_round = num_boost_round,
    evals = evals,
    verbose_eval = num_boost_round / 5 # Print information every n rounds
)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Training 3 : Early stopping

print("\n- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - ")
print("Early stopping")
model = xgb.train(
    params = params,
    dtrain = dtrain_reg,
    num_boost_round = num_boost_round,
    evals = evals,
    verbose_eval = num_boost_round / 5,
    early_stopping_rounds = early_stopping_rounds # Stop if loss does not imporove for this number of rounds
)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Training 4 : k-fold cross validation

print("\n- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - ")
print("k-fold cross validation")

# N.b. note that here I not use xgb.train() but xgb.cv()
# In this case the results is not the trained model but a pandas dataframe
# Each row of the dataframe contain the average of all splits of that round
results = xgb.cv(
    params = params,
    dtrain = dtrain_reg,
    num_boost_round = num_boost_round,
    nfold = n_fold, # If used specify the number of fold for k-fold cross validation
    early_stopping_rounds = early_stopping_rounds
)

print("\nFirst 5 lines of results dataframe: ")
print(results.head())

print("\nBest round in test set:")
idx_min = results['test-rmse-mean'].argmin()
print("{:.2f}±{:.2f} (found at row {})".format(results['test-rmse-mean'][idx_min], results['test-rmse-std'][idx_min], idx_min))

# Note that this method of cross-validation is used to see the true performance of the model. Once satisfied with its score, you must retrain it on the full data before deployment.

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Training 5 : Classification
# Until now we have trained the xgboost model for regression, specifying the objective as reg:squarederror
# To train it as a classifier we can simply change objective to binary:logistic (binary class) or multi:softprob (multiclass)

print("\n- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - ")
print("Classification")

X, y = diamonds.drop("cut", axis=1), diamonds[['cut']]

# Encode y to numeric
# This is required beacuse the xgboost requires the label to be number. So we use sklearn function to map label to number
y_encoded = OrdinalEncoder().fit_transform(y)

# Extract text features
cats = X.select_dtypes(exclude=np.number).columns.tolist()

# Convert to pd.Categorical
for col in cats: X[col] = X[col].astype('category')

# Split the data
# Stratigy specify that each set should contain (approximately) the same percentage of samples of each target class.
# Unfortunately with stratify specified some imbalance between the class cause error. So for now I will not use it
# X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, random_state = random_seed, stratify = y_encoded)
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, random_state = random_seed)

# Create classification matrices
dtrain_clf = xgb.DMatrix(X_train, y_train, enable_categorical=True)
dtest_clf = xgb.DMatrix(X_test, y_test, enable_categorical=True)

params = {"objective": "multi:softprob", "num_class": len(set(y['cut']))}
n = 1000

results = xgb.cv(
   params, dtrain_clf,
   num_boost_round = num_boost_round,
   nfold = n_fold,
   metrics = ["mlogloss", "auc", "merror"],
)

print("\nFirst 5 lines of results dataframe: ")
print(results.head())


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Training 6 : SKlearn API
# For now we have used the API provided directly by xgboost. But it also possible to use sklearn API (that sould be a wrapper around xgboost functions)
# From what I see they are the same functions but called in a way to remember sklearn workflow

print("\n- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - ")
print("Sklean API")

# Train a model using the scikit-learn API
xgb_classifier = xgb.XGBClassifier(n_estimators = 100, objective='binary:logistic', 
                                   tree_method = 'hist', eta = 0.1, max_depth = 3, enable_categorical = True
                                   )
xgb_classifier.fit(X_train, y_train)

# Convert the model to a native API model
model = xgb_classifier.get_booster()
