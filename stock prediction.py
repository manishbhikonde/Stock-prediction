import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Download historical stock data
ticker = 'AAPL'  # You can replace 'AAPL' with any stock ticker of your choice
data = yf.download(ticker, start="2015-01-01", end="2024-01-01")

# Display the first few rows of the dataset
print(data.head())



# Use the 'Close' price for prediction
data = data[['Close']]

# Create a column for the next day's Close price (target variable)
data['Prediction'] = data['Close'].shift(-1)

# Drop the last row (NaN value in 'Prediction' column)
data = data.dropna()

# Display the updated dataset
print(data.tail())



# Features (Close price)
X = np.array(data['Close']).reshape(-1, 1)

# Target (Next day's Close price)
y = np.array(data['Prediction'])

# Split the data into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Initialize the Linear Regression model
model = LinearRegression()

# Train the model on the training data
model.fit(X_train, y_train)


# Predict the stock prices for the test data
y_pred = model.predict(X_test)

# Evaluate the model's performance
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")



# Plot the actual vs predicted prices
plt.figure(figsize=(10, 6))
plt.plot(y_test, label="Actual Price")
plt.plot(y_pred, label="Predicted Price")
plt.title(f"{ticker} Stock Price Prediction")
plt.xlabel('Days')
plt.ylabel('Close Price')
plt.legend()
plt.show()
