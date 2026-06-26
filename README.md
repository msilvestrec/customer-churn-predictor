# Customer Churn Predictor

ML model that predicts whether a telecom customer will churn, deployed as a REST API.

## Live API
https://customer-churn-predictor-production-9f46.up.railway.app

## Documentation
https://customer-churn-predictor-production-9f46.up.railway.app/docs

## Tech Stack
- XGBoost + scikit-learn pipeline
- FastAPI
- Docker
- Railway

## Project Structure
- `notebooks/` - EDA, data preparation and SHAP explainability
- `src/train.py` - Training script
- `api/main.py` - REST API
- `Dockerfile` - Container configuration

## Setup
```bash
git clone https://github.com/msilvestrec/customer-churn-predictor.git
cd customer-churn-predictor
pyenv virtualenv 3.10.6 customer-churn-predictor
pyenv local customer-churn-predictor
pip install -r requirements.txt
python src/train.py
uvicorn api.main:app --reload
```

## Dataset
[Telco Customer Churn - Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
