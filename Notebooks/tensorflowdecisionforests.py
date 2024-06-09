# -*- coding: utf-8 -*-
"""TensorFlowDecisionForests.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1KIFCqfWfZkhbAALlbA7KeQVtZSjjm3H1
"""

# !pip install tensorflow_decision_forests

# https://www.tensorflow.org/decision_forests/api_docs/python/tfdf/keras

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow_decision_forests as tfdf
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, r2_score

# Load the dataset
file_path = 'Carbon Emission.csv'
df = pd.read_csv(file_path)

# Check loading worked
print(df.head())
print(df.info())

# Check for missing values
print(df.isnull().sum())

# Based on previous work we know Vehicle has null values
nullVehicleType = df.loc[df['Vehicle Type'].isnull()]
print(nullVehicleType['Transport'].unique())

# Updating Vehicle
dfNonNull = df.fillna('None')
print(dfNonNull.isnull().sum())

# Display the first few rows after handling missing values
print(dfNonNull.head())

# I also could not get pipeline to play nice with random forests. Re-used this from KNN
# Map each categorical column to numeric values
dfNonNull['Body Type'] = dfNonNull['Body Type'].map({'overweight': 0, 'obese': 1, 'underweight': 2, 'normal': 3})
dfNonNull['Sex'] = dfNonNull['Sex'].map({'female': 0, 'male': 1})
dfNonNull['Diet'] = dfNonNull['Diet'].map({'pescatarian': 0, 'vegetarian': 1, 'omnivore': 2, 'vegan': 3})
dfNonNull['How Often Shower'] = dfNonNull['How Often Shower'].map({'daily': 0, 'less frequently': 1, 'more frequently': 2, 'twice a day': 3})
dfNonNull['Heating Energy Source'] = dfNonNull['Heating Energy Source'].map({'coal': 0, 'natural gas': 1, 'wood': 2, 'electricity': 3})
dfNonNull['Transport'] = dfNonNull['Transport'].map({'public': 0, 'walk/bicycle': 1, 'private': 2})
dfNonNull['Vehicle Type'] = dfNonNull['Vehicle Type'].map({'None': 0, 'petrol': 1, 'diesel': 2, 'hybrid': 3, 'lpg': 4, 'electric': 5})
dfNonNull['Social Activity'] = dfNonNull['Social Activity'].map({'often': 0, 'never': 1, 'sometimes': 2})
dfNonNull['Frequency of Traveling by Air'] = dfNonNull['Frequency of Traveling by Air'].map({'frequently': 0, 'rarely': 1, 'never': 2, 'very frequently': 3})
dfNonNull['Waste Bag Size'] = dfNonNull['Waste Bag Size'].map({'large': 0, 'extra large': 1, 'small': 2, 'medium': 3})
dfNonNull['Energy efficiency'] = dfNonNull['Energy efficiency'].map({'No': 0, 'Sometimes': 1, 'Yes': 2})

# Display the dataframe info after encoding
print(dfNonNull.info())
print(dfNonNull.head())

# Drop recycling and cooking
dfNonNull = dfNonNull.drop('Recycling', axis=1)
dfNonNull = dfNonNull.drop('Cooking_With', axis=1)
print(dfNonNull.head())

# Split into features and trget
X = dfNonNull.drop(columns=['CarbonEmission'])
y = dfNonNull['CarbonEmission']

# Using test train split here but I am thinking there is a better way
# the TFDF model does not play nice with these datasets directly.

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# tfdfd does not simply work with the pandas dataframe that train test split retuns
# we are making a copy of the dataframes and adding data frame and adding the target column to them
# This process feels like this could be simplified somehow
train_df = X_train.copy()
train_df['CarbonEmission'] = y_train

# We have to do something similar with the test set
test_df = X_test.copy()
test_df['CarbonEmission'] = y_test

# CReate the Model
model = tfdf.keras.RandomForestModel(task=tfdf.keras.Task.REGRESSION)

# Converting to the dataset that tfdf uses
train_ds = tfdf.keras.pd_dataframe_to_tf_dataset(train_df, label="CarbonEmission", task=tfdf.keras.Task.REGRESSION)

# Fit the model on the training data
model.fit(train_ds)

# For the testing data, we also have to convert to a format that the model can use
test_ds = tfdf.keras.pd_dataframe_to_tf_dataset(test_df, label="CarbonEmission", task=tfdf.keras.Task.REGRESSION)

# Here we are getting the most important features
most_important_features = model.make_inspector().variable_importances()["NUM_AS_ROOT"]

# Print the ffeatures
print("Most important features:")
for attribute, importance in most_important_features:
    print(f"{attribute}: {importance}")

# Step 11: Test the model on multiple random rows
num_tests = 10  # Number of random rows to test
for _ in range(num_tests):
    # Select a random row from the dataset
    random_row = X.sample()

    # Convert the random row to a TensorFlow dataset
    random_row_ds = tfdf.keras.pd_dataframe_to_tf_dataset(random_row, task=tfdf.keras.Task.REGRESSION)

    # Predict the carbon emission for the selected row
    random_row_pred = model.predict(random_row_ds)
    print(f'Predicted Carbon Emission: {random_row_pred[0][0]}')

    # Get the actual carbon emission for the selected row
    random_row_actual = dfNonNull.loc[random_row.index[0], 'CarbonEmission']
    print(f'Actual Carbon Emission: {random_row_actual}')
    print()

# Lets test the model on the entire test set
y_pred = model.predict(test_ds)
y_pred = np.array(y_pred)
y_pred = y_pred.flatten()

# Calculate the Mean Absolute Error (MAE) and R^2 score
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Absolute Error: {mae}")
print(f"R^2 Score: {r2}")


# these saves are not working

# # Save the model
# model.save("carbon_emission_model")

# # https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/keras/save_and_load.ipynb#scrollTo=m2dkmJVCGUia

# # Save the entire model to a HDF5 file.
# # The '.h5' extension indicates that the model should be saved to HDF5.
# model.save('my_model.h5')

model.summary()

# save this model

model.save("my_model.keras")

# # Load the model

# # Recreate the exact same model, including its weights and the optimizer
new_model = tf.keras.models.load_model('my_model.keras')

new_model.summary()

## I am getting this error:
#Expected object to be an instance of `KerasSaveable`, but got <tensorflow_decision_forests.keras.RandomForestModel object at 0x16fef88d0> of type <class 'tensorflow_decision_forests.keras.RandomForestModel'>