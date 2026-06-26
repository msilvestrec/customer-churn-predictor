import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer
import xgboost as xgb
import joblib

def train():
    df = pd.read_csv('data/WA_Fn-UseC_-Telco-Customer-Churn.csv')
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)
    df = df.drop(columns='customerID')

    X = df.drop(columns='Churn')
    y = df['Churn'].map({'Yes': 1, 'No': 0})

    categorical_cols = X.select_dtypes(include='object').columns.tolist()
    numerical_cols = X.select_dtypes(exclude='object').columns.tolist()

    preprocessor = ColumnTransformer(transformers=[
        ('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), categorical_cols),
        ('num', 'passthrough', numerical_cols)
    ])

    scale = y.value_counts()[0] / y.value_counts()[1]

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', xgb.XGBClassifier(random_state=42, eval_metric='logloss', scale_pos_weight=scale))
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    pipeline.fit(X_train, y_train)

    joblib.dump(pipeline, 'models/model.pkl')

if __name__ == '__main__':
    train()
