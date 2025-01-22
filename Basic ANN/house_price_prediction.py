import pandas as pd
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error,r2_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Load the data
data = pd.read_csv('D:/NN/housing price/data.csv')
X=pd.get_dummies(columns = ['mainroad', 'guestroom', 'basement','furnishingstatus', 'hotwaterheating', 'airconditioning', 'prefarea'], data=data.drop(columns=['price']))
y=data['price']

# Normalization
numerical_cols = ['area', 'bedrooms', 'bathrooms', 'stories', 'parking']

scaler = StandardScaler()
X[numerical_cols] = scaler.fit_transform(X[numerical_cols])



# Normalization for output also
scaler = StandardScaler()
y_scaled = scaler.fit_transform(y.values.reshape(-1, 1))

y = pd.Series(y_scaled.flatten(), index=y.index, name=y.name)



# Building model
model = Sequential()
model.add(Input(shape=(X.shape[1],)))
model.add(Dense(512, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1,activation='sigmoid'))

model.summary()

# compile and training
model.compile(loss='mean_squared_error',
              optimizer='adam',
              metrics=['mae', 'mse'])

model.fit(X, y, epochs=100, validation_split=0.2)




# testing
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Predict prices
y_pred = model.predict(X)

# Calculate error metrics
print(f"MAE: {mean_absolute_error(y, y_pred):,.2f}")
print(f"MSE: {mean_squared_error(y, y_pred):,.2f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y, y_pred)):,.2f}")
print(f"R² Score: {r2_score(y, y_pred):.4f}")
