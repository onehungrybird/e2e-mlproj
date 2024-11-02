# src/prediction/predict.py

import pandas as pd
import pickle
from src.data_prep.data_utils import load_data, preprocess_data
from sklearn.metrics import mean_squared_error, r2_score
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load the trained model
with open('src/model/sales_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Load the test data
X_test = pd.read_csv('data/X_test.csv')
y_test = pd.read_csv('data/y_test.csv')

# Make predictions
predictions = model.predict(X_test)
print(predictions, '\n')

# Calculate evaluation metrics
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print(f"Model evaluation on test data:\nMean Squared Error: {mse}\nR-squared: {r2}")

# Print predictions
for sales in predictions:
    logging.info(f"Predicted Sales: {sales:.2f}")