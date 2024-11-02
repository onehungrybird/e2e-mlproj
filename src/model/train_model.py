# src/model/train_model.py

import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression
from src.data_prep.data_utils import load_data, preprocess_data
from sklearn.model_selection import train_test_split

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

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Save the model
with open('src/model/sales_model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

print("Model trained and saved successfully.")
