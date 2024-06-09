<<<<<<< Updated upstream
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
=======
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np

app = Flask(__name__)
CORS(app, resources={r"/predict": {"origins": "http://localhost:5173"}})  # Allow only the React app origin

model = joblib.load('best_gradient_boosting_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    
    input_data = np.array([
        data['bodyType'],
        data['sex'],
        data['diet'],
        data['howOftenShower'],
        data['heatingEnergySource'],
        data['transport'],
        data['vehicleType'],
        data['socialActivity'],
        data['frequencyOfTravelingByAir'],
        data['wasteBagSize'],
        data['energyEfficiency'],
        float(data['monthlyGroceryBill']),
        float(data['vehicleMonthlyDistanceKm']),
        float(data['wasteBagWeeklyCount']),
        float(data['howLongTVPCDailyHour']),
        float(data['howManyNewClothesMonthly']),
        float(data['howLongInternetDailyHour'])
    ]).reshape(1, -1)
    
    prediction = model.predict(input_data)
    return jsonify({'predictedEmission': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True, port=5001)
>>>>>>> Stashed changes
