import numpy as np
import pandas as pd

from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler 
from sklearn.metrics import accuracy_score

data = pd.read_csv("D:/NN/bike_demand/data.csv", encoding='latin-1')
X= data.drop(["Date","Rented Bike Count","Seasons","Holiday","Functioning Day"])
Y=data["Rented Bike Count"]

xtrain,xtest,ytrain,ytest = train_test_split(X,Y,test_size=0.1,random_state=1)

j=MinMaxScaler()

xtrain = j.fit_transform(xtrain)
xtest= j.fit_transform(xtest)

# k = Sequential()
# k.add(Dense(100,activation = 'relu',input_shape = (9,)))
# k.add(Dense(1,activation = 'linear'))


# print(k.summary())