import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
import joblib

# Load the dataset
file_path = 'Carbon Emission.csv'
df = pd.read_csv(file_path)
print(df.head())

# Define preprocessing steps for numerical and categorical features
numerical_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = ['Body Type', 'Sex', 'Diet', 'How Often Shower', 'Heating Energy Source', 'Transport', 'Vehicle Type', 'Social Activity', 'Frequency of Traveling by Air', 'Waste Bag Size', 'Energy efficiency']

# Drop target and unnecessary columns from numerical features
numerical_features.remove('CarbonEmission')
numerical_features = [feature for feature in numerical_features if feature not in ['Recycling', 'Cooking_With']]

# Define the preprocessing pipeline for numerical and categorical features
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine the preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Define the full pipeline including preprocessing and the model
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('poly', PolynomialFeatures(degree=2, interaction_only=True)),
    ('regressor', GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42))
])

# Split the data
X = df.drop(columns=['CarbonEmission'])
y = df['CarbonEmission']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the pipeline
pipeline.fit(X_train, y_train)

# Make predictions
y_pred = pipeline.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Absolute Error: {mae}")
print(f"R^2 Score: {r2}")

# Save the pipeline
joblib.dump(pipeline, 'carbon_emission_pipeline.pkl')

# Load the pipeline
loaded_pipeline = joblib.load('carbon_emission_pipeline.pkl')

# Verify the loaded pipeline
y_pred = loaded_pipeline.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Loaded Pipeline Mean Absolute Error: {mae}")
print(f"Loaded Pipeline R^2 Score: {r2}")

# Make predictions with the loaded pipeline
for i in range(10):
    # Select a random row from the test set
    random_index = np.random.randint(0, X_test.shape[0])
    random_sample = X_test.iloc[random_index].values.reshape(1, -1)
    random_sample_df = pd.DataFrame(random_sample, columns=X_test.columns)

    # Predict the carbon emission for the selected sample
    predicted_emission = loaded_pipeline.predict(random_sample_df)[0]
    actual_emission = y_test.iloc[random_index]

    # Print the results
    print(f"Sample {i+1}:")
    print(f"Predicted Carbon Emission: {predicted_emission}")
    print(f"Actual Carbon Emission: {actual_emission}")
    print()
