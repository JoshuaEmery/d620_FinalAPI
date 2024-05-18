import pandas as pd
import json

# Load the CSV file
df = pd.read_csv('Carbon Emission.csv')

# Get the columns and exclude 'CarbonEmission'
columns = df.columns.drop('CarbonEmission')

# Initialize a dictionary to store distinct values
distinct_values = {}

# Loop through columns, convert data types and store unique values
for column in columns:
    # Using map() to convert all values to Python native types for JSON serialization
    distinct_values[column] = list(df[column].dropna().unique().astype(str))

# Print the dictionary to check the output
print(distinct_values)

# Write the dictionary to a JSON file
with open('Carbon Emission.json', 'w') as f:
    json.dump(distinct_values, f, indent=4)
