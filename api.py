import tensorflow as tf
from flask import Flask
import joblib
import pandas as pd
from flask import jsonify
from KN import KNpredict

# Load the model from disk
clf_loaded = joblib.load('./Notebooks/models/KNRegression.pkl')

app = Flask(__name__)


@app.route("/KNpredict")
def predict():
    # lets pull a random person from Carbon Emission.csv
    df = pd.read_csv('./Notebooks/models/Carbon Emission.csv')
    x = df.drop(columns=['CarbonEmission'])
    y = df['CarbonEmission']
    person = x.sample()
    # use the model to predict the carbon emission
    prediction = KNpredict(person)  # actual carbon emission
    actual_carbon_emission = y.loc[person.index]
    print(f'Actual Carbon Emission: {actual_carbon_emission.values[0]}')
    # return a json object with the person, predicted carbon emission, and actual carbon emission
    return jsonify({
        'Person': person.to_dict(),
        'Predicted Carbon Emission': str(prediction),
        'Actual Carbon Emission': str(actual_carbon_emission.values[0])
    })

@app.route("/")
def greeting():
    return "DS620 Emissions API"

@app.route("/testjson")
def testjson():
    return jsonify({
        'Person': {
            'Age': 25,
            'Income': 50000,
            'HouseholdSize': 2,
            'MilesFromCity': 10
        },
        'Predicted Carbon Emission': 3.5,
        'Actual Carbon Emission': 3.2
    })

# We need a method that will take in a person from req.body and return a datframe of that person to use with the model
def person_to_df(person):
    # 
    return pd.DataFrame([person])


if __name__ == "__main__":
    app.run()
