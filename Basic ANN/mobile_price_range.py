import pandas as pd
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler


data = pd.read_csv('D:/NN/mobilePriceRange/train.csv')
# data = data.drop(columns='id')
x=data.drop(columns='price_range')
y=data['price_range']

xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.1,random_state=1)

# print(data['price_range'].unique())

j=MinMaxScaler()
xtrain=j.fit_transform(xtrain)
xtest=j.fit_transform(xtest)


model=Sequential()

model.add(Dense(120,activation='relu',input_shape=(20,)))
model.add(Dense(240,activation='relu'))
model.add(Dense(80,activation='relu'))
model.add(Dense(4,activation='softmax'))


print(model.summary())

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

model.fit(xtrain,ytrain,epochs=100,validation_split=0.1)
