# This librar loads the KNRegression model and uses it to predict the carbon emission of a person
# For this model ALL Fields are required. The person is expected to be a DataFrame
import joblib
def KNpredict(person):
    # Load the model from disk
    clf_loaded = joblib.load('./Notebooks/models/KNRegression.pkl')
    # use the model to predict the carbon emission
    predicted_carbon_emission = clf_loaded.predict(person)
    return predicted_carbon_emission[0]
