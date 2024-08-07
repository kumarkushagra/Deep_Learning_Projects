import pandas as pd
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error,r2_score
from sklearn.preprocessing import MinMaxScaler


# Chance of Admit 
data = pd.read_csv("D:/NN/graduate admission dataset/Admission_Predict_Ver1.1.csv")
data=data.drop(columns="Serial No.")

x=data.drop(columns="Chance of Admit ")
y=data["Chance of Admit "]

xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.05,random_state=1)

j=MinMaxScaler()

xtrain=j.fit_transform(xtrain)
xtest=j.fit_transform(xtest)




model = Sequential()

model.add(Dense(7,activation='relu',input_shape=(7,)))
model.add(Dense(30,activation='relu'))
model.add(Dense(30,activation='relu'))
model.add(Dense(1,activation='linear'))

print(model.summary())

model.compile(loss='mean_squared_error',optimizer='adam')
model.fit(xtrain,ytrain,epochs=100,validation_split=0.2)


ypred = model.predict(xtest)
mae = mean_absolute_error(ytest, ypred)
print("Mean Absolute Error:", mae)
print(r2_score(ypred,ytest))