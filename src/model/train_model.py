# src/model/train_model.py

import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression
from src.data_prep.data_utils import load_data, preprocess_data
from sklearn.model_selection import train_test_split
import logging
import mlflow
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
import os, sys

# Add the 'src' directory to PYTHONPATH
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load and preprocess data
data = load_data('data/sales_data.csv')
data = preprocess_data(data)

# Feature engineering
data['Day'] = data['Date'].dt.day
data['Month'] = data['Date'].dt.month
X = data[['Store', 'Day', 'Month']]
y = data['Sales']

# One-hot encoding for categorical features
X = pd.get_dummies(X, columns=['Store'], drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2)

# Save test data for later use in prediction.py
X_test.to_csv('data/X_test.csv', index=False)
y_test.to_csv('data/y_test.csv', index=False)

params = {
    "n_estimators": 50
}

# Set MLflow tracking URI to a local directory within the project
# mlflow.set_tracking_uri("file://" + os.path.join(os.getcwd(), "mlruns"))
# mlflow.set_tracking_uri(r"file:///C:/Users/sahuman/Downloads/mlproj/mlruns")
# mlflow.set_tracking_uri("file:///C:/Users/sahuman/Downloads/mlproj/mlruns")

tracking_uri = os.path.join(".", "mlruns")
os.makedirs(tracking_uri, exist_ok=True)
mlflow.set_tracking_uri(tracking_uri)

# Create a new MLflow Experiment
mlflow.set_experiment("MLflow Quickstart")

with mlflow.start_run():
    # Train the model
    # model = LinearRegression()
    model = RandomForestRegressor(**params)
    model.fit(X_train, y_train)
    
    pred = model.predict(X_test)
    mse = mean_squared_error(y_test, pred)
    r2 = r2_score(y_test, pred)
    mae = mean_absolute_error(y_test, pred)
    
    mlflow.log_params(params)
    mlflow.log_metric('mse', mse)
    mlflow.log_metric('r2', r2)
    mlflow.log_metric('mae', mae)
    mlflow.sklearn.log_model(model, 'sales_forecasting_model')

# Save the model
with open('src/model/sales_model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

logging.info("Model trained and saved successfully.")
