import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xgb
import joblib

def train():
    df = pd.read_csv('data/churn_clean.csv')

    X = df.drop(columns='Churn')
    y = df['Churn']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scale = y_train.value_counts()[0] / y_train.value_counts()[1]

    model = xgb.XGBClassifier(random_state=42, eval_metric='logloss', scale_pos_weight=scale)
    model.fit(X_train, y_train)

    joblib.dump(model, 'models/model.pkl')
    print('Modelo guardado en models/model.pkl')

if __name__ == '__main__':
    train()
