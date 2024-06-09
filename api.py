from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI()

# Allow all origins, in a production environment, specify allowed origins
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictionRequest(BaseModel):
    bodyType: str
    sex: str
    diet: str
    howOftenShower: str
    heatingEnergySource: str
    transport: str
    vehicleType: str
    socialActivity: str
    frequencyOfTravelingByAir: str
    wasteBagSize: str
    energyEfficiency: str
    monthlyGroceryBill: float
    vehicleMonthlyDistanceKm: float
    wasteBagWeeklyCount: float
    howLongTVPCDailyHour: float
    howManyNewClothesMonthly: float
    howLongInternetDailyHour: float

# Load the trained pipeline
pipeline = joblib.load('carbon_emission_pipeline.pkl')

@app.get("/")
def read_root():
    return {"message": "API is working!"}

@app.post("/predict")
async def predict(request: PredictionRequest):
    # Convert input data to DataFrame
    input_data = pd.DataFrame([{
        "Body Type": request.bodyType,
        "Sex": request.sex,
        "Diet": request.diet,
        "How Often Shower": request.howOftenShower,
        "Heating Energy Source": request.heatingEnergySource,
        "Transport": request.transport,
        "Vehicle Type": request.vehicleType,
        "Social Activity": request.socialActivity,
        "Frequency of Traveling by Air": request.frequencyOfTravelingByAir,
        "Waste Bag Size": request.wasteBagSize,
        "Energy efficiency": request.energyEfficiency,
        "Monthly Grocery Bill": request.monthlyGroceryBill,
        "Vehicle Monthly Distance Km": request.vehicleMonthlyDistanceKm,
        "Waste Bag Weekly Count": request.wasteBagWeeklyCount,
        "How Long TV PC Daily Hour": request.howLongTVPCDailyHour,
        "How Many New Clothes Monthly": request.howManyNewClothesMonthly,
        "How Long Internet Daily Hour": request.howLongInternetDailyHour
    }])
    print(input_data.head())
    # Make prediction
    prediction = pipeline.predict(input_data)
    return {"predictedEmission": prediction[0]}

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)
