import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV
import joblib

# Load the dataset
file_path = 'Carbon Emission.csv'
df = pd.read_csv(file_path)

# Fill missing values
dfNonNull = df.fillna('None')

# Define categorical and numerical columns
categorical_columns = ['Body Type', 'Sex', 'Diet', 'How Often Shower', 'Heating Energy Source', 'Transport', 'Vehicle Type', 'Social Activity', 'Frequency of Traveling by Air', 'Waste Bag Size', 'Energy efficiency']
numerical_columns = ['Monthly Grocery Bill', 'Vehicle Monthly Distance (km)', 'Waste Bag Weekly Count', 'How Long TV/PC Daily (hour)', 'How Many New Clothes Monthly', 'How Long Internet Daily (hour)']

# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(), categorical_columns),
        ('num', StandardScaler(), numerical_columns)
    ]
)

# Complete pipeline with model
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('poly', PolynomialFeatures(degree=2, interaction_only=True)),
    ('scaler', StandardScaler()),
    ('model', GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42))
])

# Split into features and target
X = dfNonNull.drop(columns=['CarbonEmission'])
y = dfNonNull['CarbonEmission']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
pipeline.fit(X_train, y_train)

# Make predictions
y_pred = pipeline.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Absolute Error: {mae}")
print(f"R^2 Score: {r2}")

# Save the pipeline
joblib.dump(pipeline, 'carbon_emission_model_pipeline.pkl')

# Load the pipeline
loaded_pipeline = joblib.load('carbon_emission_model_pipeline.pkl')

# Verify the loaded pipeline
y_pred_loaded = loaded_pipeline.predict(X_test)
mae_loaded = mean_absolute_error(y_test, y_pred_loaded)
r2_loaded = r2_score(y_test, y_pred_loaded)
print(f"Loaded Model Mean Absolute Error: {mae_loaded}")
print(f"Loaded Model R^2 Score: {r2_loaded}")

# Make predictions with the best loaded model
for i in range(10):
    # Select a random row from the test set
    random_index = np.random.randint(0, X_test.shape[0])
    random_sample = X_test[random_index].reshape(1, -1)

    # Predict the carbon emission for the selected sample
    predicted_emission = loaded_pipeline.predict(random_sample)[0]
    actual_emission = y_test.iloc[random_index]

    # Print the results
    print(f"Sample {i+1}:")
    print(f"Predicted Carbon Emission: {predicted_emission}")
    print(f"Actual Carbon Emission: {actual_emission}")
    print()
