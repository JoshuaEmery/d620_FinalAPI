import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import mean_absolute_error, r2_score

# Load the dataset
file_path = 'Carbon Emission.csv'
df = pd.read_csv(file_path)

# Check loading worked
print(df.head())
print(df.info())

# Check for missing values
print(df.isnull().sum())

# Updating Vehicle
dfNonNull = df.fillna('None')
print(dfNonNull.isnull().sum())

# Display the first few rows after handling missing values
print(dfNonNull.head())

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

# Drop recycling and cooking
dfNonNull = dfNonNull.drop('Recycling', axis=1)
dfNonNull = dfNonNull.drop('Cooking_With', axis=1)
print(dfNonNull.head())

# Split into features and target
X = dfNonNull.drop(columns=['CarbonEmission'])
y = dfNonNull['CarbonEmission']

# Using test train split here but I am thinking there is a better way
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Keras Sequential Model
model = Sequential([
    Dense(64, activation='relu', input_shape=[X_train.shape[1]]),
    Dense(64, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Train the model
model.fit(X_train, y_train, epochs=50, validation_split=0.2, verbose=1)

# Evaluate the model
loss, mae = model.evaluate(X_test, y_test, verbose=1)
print(f"Mean Absolute Error on test set: {mae}")

# Save the model
model.save("carbon_emission_model.keras")

# Load the model
new_model = tf.keras.models.load_model('carbon_emission_model.keras')

# Verify the loaded model
new_model.summary()

# Test the model on multiple random rows
num_tests = 10  # Number of random rows to test
for _ in range(num_tests):
    # Select a random row from the dataset
    random_row = X.sample()

    # Predict the carbon emission for the selected row
    random_row_pred = new_model.predict(random_row)
    print(f'Predicted Carbon Emission: {random_row_pred[0][0]}')

    # Get the actual carbon emission for the selected row
    random_row_actual = dfNonNull.loc[random_row.index[0], 'CarbonEmission']
    print(f'Actual Carbon Emission: {random_row_actual}')
    print()

# Make predictions on the entire test set
y_pred = new_model.predict(X_test)
y_pred = y_pred.flatten()

# Calculate the Mean Absolute Error (MAE) and R^2 score
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Absolute Error: {mae}")
print(f"R^2 Score: {r2}")
