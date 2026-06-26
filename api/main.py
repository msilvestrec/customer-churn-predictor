from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI()

model = joblib.load('models/model.pkl')

class Customer(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float

@app.get('/')
def root():
    return {'message': 'Churn Predictor API'}

@app.post('/predict')
def predict(customer: Customer):
    data = pd.DataFrame([customer.dict()])
    prediction = model.predict(data)[0]
    probability = model.predict_proba(data)[0][1]
    return {
        'churn': bool(prediction),
        'probability': round(float(probability), 2)
    }
