from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np

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

model = joblib.load('best_gradient_boosting_model.pkl')

@app.get("/")
def read_root():
    return {"message": "API is working!"}

@app.post("/predict")
async def predict(request: PredictionRequest):
    input_data = np.array([
        request.bodyType,
        request.sex,
        request.diet,
        request.howOftenShower,
        request.heatingEnergySource,
        request.transport,
        request.vehicleType,
        request.socialActivity,
        request.frequencyOfTravelingByAir,
        request.wasteBagSize,
        request.energyEfficiency,
        request.monthlyGroceryBill,
        request.vehicleMonthlyDistanceKm,
        request.wasteBagWeeklyCount,
        request.howLongTVPCDailyHour,
        request.howManyNewClothesMonthly,
        request.howLongInternetDailyHour
    ]).reshape(1, -1)
    
    prediction = model.predict(input_data)
    return {"predictedEmission": prediction[0]}

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)
