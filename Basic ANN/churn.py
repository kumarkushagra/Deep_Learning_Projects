import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential,Input,regularizers
from tensorflow.keras.layers import Dense

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data= pd.read_csv("D:/NN/churn modeling/churn.csv")

y=data["Exited"]
X = pd.get_dummies(data.drop(columns=["RowNumber", "CustomerId", "Surname", "Exited"]),
                   columns=['Gender', 'Geography'])
X = X.astype(int)

xtrain,xtest,ytrain,ytest = train_test_split(X,y,test_size=0.2,random_state=1)

j=MinMaxScaler()

xtrain = j.fit_transform(xtrain)
xtest = j.fit_transform(xtest)

model = Sequential()
model.add(Input(shape=(X.shape[1],)))
model.add(Dense(100, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
model.add(Dense(260, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
model.add(Dense(720,activation='relu',  kernel_regularizer=regularizers.l2(0.001)))
model.add(Dense(1200,activation='relu',  kernel_regularizer=regularizers.l2(0.001)))
model.add(Dense(1200,activation='relu',  kernel_regularizer=regularizers.l2(0.001)))
model.add(Dense(1700,activation='relu',  kernel_regularizer=regularizers.l2(0.001)))
model.add(Dense(1200,activation='relu',  kernel_regularizer=regularizers.l2(0.001)))
model.add(Dense(1200,activation='relu',  kernel_regularizer=regularizers.l2(0.001)))
model.add(Dense(900,activation='relu',  kernel_regularizer=regularizers.l2(0.001)))
model.add(Dense(500,activation='relu',  kernel_regularizer=regularizers.l2(0.001)))
model.add(Dense(320,activation='relu',  kernel_regularizer=regularizers.l2(0.001)))
model.add(Dense(260,activation='relu',  kernel_regularizer=regularizers.l2(0.001)))
model.add(Dense(120,activation='relu',  kernel_regularizer=regularizers.l2(0.001)))
model.add(Dense(1,activation='sigmoid'))

print(model.summary())

model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              optimizer = 'adam',metrics=["accuracy"])
model.fit(xtrain,ytrain,epochs=100,validation_split=0.1)

yprob = model.predict(xtest)
ypred= (yprob > 0.5).astype(int)
j=accuracy_score(ytest,ypred)
print("ACCURACY: ",j*100)
