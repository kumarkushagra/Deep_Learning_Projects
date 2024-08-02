import numpy as np
import pandas as pd
# from tensorflow import keras
# from tensorflow.keras import Sequential
# from tensorflow.keras.layers import Dense
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_squared_error

# Load the data
data = pd.read_csv('D:/NN/housing price/data.csv')
# data = data.drop(columns=["date", "yr_renovated", "street", "city", "statezip", "country"])


print(data['city'].unique())





# # Separate the target variable from the features
# y = data["price"]
# X = data.drop(columns="price")

# # Scale the features and target variable
# scaler = MinMaxScaler()
# X_scaled = scaler.fit_transform(X)
# y_scaled = scaler.fit_transform(y.values.reshape(-1, 1)).flatten()

# # Split the data into training and test sets
# X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=1)

# # Define the neural network model
# model = Sequential()
# model.add(Dense(144, activation='relu', input_shape=(X.shape[1],)))
# model.add(Dense(128, activation='relu'))
# model.add(Dense(256, activation='relu'))
# model.add(Dense(128, activation='relu'))
# model.add(Dense(1, activation='linear'))

# # Compile the model
# model.compile(loss='mean_squared_error', optimizer='adam')

# # Train the model
# model.fit(X_train, y_train, epochs=50, validation_split=0.1)

# # Make predictions
# y_pred = model.predict(X_test)

# # Calculate mean squared error
# mse = mean_squared_error(y_test, y_pred)
# print("Mean Squared Error on Test Set:", mse)

# # Inverse transform to get actual prices
# y_pred_actual = scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
# y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

# # Print the first predicted and actual prices
# print("Predicted Price for the first house:", y_pred_actual[3])
# print("Actual Price for the first house:", y_test_actual[3])
