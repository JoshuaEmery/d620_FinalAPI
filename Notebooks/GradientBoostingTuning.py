import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV
import joblib

# Load the dataset
file_path = 'Carbon Emission.csv'
df = pd.read_csv(file_path)

# Fill missing values
dfNonNull = df.fillna('None')

# Encode categorical columns
categorical_columns = ['Body Type', 'Sex', 'Diet', 'How Often Shower', 'Heating Energy Source', 'Transport', 'Vehicle Type', 'Social Activity', 'Frequency of Traveling by Air', 'Waste Bag Size', 'Energy efficiency']
dfNonNull = pd.get_dummies(dfNonNull, columns=categorical_columns)

# Drop unnecessary columns
dfNonNull = dfNonNull.drop(['Recycling', 'Cooking_With'], axis=1)

# Split into features and target
X = dfNonNull.drop(columns=['CarbonEmission'])
y = dfNonNull['CarbonEmission']

# Create polynomial features
poly = PolynomialFeatures(degree=2, interaction_only=True)
X_poly = poly.fit_transform(X)

# Normalize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_poly)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Comprehensive hyperparameter tuning for Gradient Boosting model
param_grid = {
    'n_estimators': [300],
    'learning_rate': [0.1],
    'max_depth': [3],
    'min_samples_split': [2, 5, 10, 15],
    'min_samples_leaf': [1, 2, 4, 6],
    'subsample': [0.6, 0.8, 0.9, 1.0],
    'max_features': ['sqrt', 'log2', None]
}

# Initialize GridSearchCV with verbose set to 3
grid_search = GridSearchCV(GradientBoostingRegressor(random_state=42), param_grid, cv=5, scoring='r2', n_jobs=-1, verbose=3)
grid_search.fit(X_train, y_train)

# Get the best model
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

# Evaluate the best model
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Best Model Mean Absolute Error: {mae}")
print(f"Best Model R^2 Score: {r2}")
# Show the best hyperparameters
print(grid_search.best_params_)
# Show summary of the best model
print(best_model)

# Save the best model
joblib.dump(best_model, 'best_gradient_boosting_model.pkl')

# Load the best model
loaded_model = joblib.load('best_gradient_boosting_model.pkl')

# Verify the loaded model
y_pred = loaded_model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Loaded Model Mean Absolute Error: {mae}")
print(f"Loaded Model R^2 Score: {r2}")

# Make predictions with the best loaded model
for i in range(10):
    # Select a random row from the test set
    random_index = np.random.randint(0, X_test.shape[0])
    random_sample = X_test[random_index].reshape(1, -1)

    # Predict the carbon emission for the selected sample
    predicted_emission = loaded_model.predict(random_sample)[0]
    actual_emission = y_test.iloc[random_index]

    # Print the results
    print(f"Sample {i+1}:")
    print(f"Predicted Carbon Emission: {predicted_emission}")
    print(f"Actual Carbon Emission: {actual_emission}")
    print()
