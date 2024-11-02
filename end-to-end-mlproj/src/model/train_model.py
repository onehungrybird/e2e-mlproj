# src/model/train_model.py

import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression
from src.data_preprocessing.data_utils import load_data, preprocess_data

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

# Train the model
model = LinearRegression()
model.fit(X, y)

# Save the model
with open('models/sales_model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

print("Model trained and saved successfully.")
