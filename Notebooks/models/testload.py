import joblib
import pandas as pd
# Load the model from disk
clf_loaded = joblib.load('KNRegression.pkl')

# lets pull a random person from Carbon Emission.csv
df = pd.read_csv('Carbon Emission.csv')
x = df.drop(columns=['CarbonEmission'])
y = df['CarbonEmission']
person = x.sample()
# use the model to predict the carbon emission
predicted_carbon_emission = clf_loaded.predict(person)
print(f'Person: {person}')
print(f'Predicted Carbon Emission: {predicted_carbon_emission[0]}')
# actual carbon emission
actual_carbon_emission = y.loc[person.index]
print(f'Actual Carbon Emission: {actual_carbon_emission.values[0]}')