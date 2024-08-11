#96% accuracy

import pandas as pd
from tensorflow import keras 
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler

data = pd.read_csv('D:/NN/mushroom dataset/mushroom_cleaned.csv')

X=data.drop(columns='class')
Y=data['class']

x_train,x_test,y_train,y_test  = train_test_split(X,Y,test_size=0.2,random_state=1)

k=MinMaxScaler()
x_train=k.fit_transform(x_train)
x_test=k.fit_transform(x_test)

model = Sequential([Dense(30,activation='relu',input_shape=(8,))])
model.add(Dense(80,activation='relu'))
model.add(Dense(120,activation='relu'))
model.add(Dense(120,activation='relu'))
model.add(Dense(120,activation='relu'))
model.add(Dense(80,activation='relu'))
model.add(Dense(1,activation='sigmoid'))

print(model.summary())

model.compile(loss='mean_squared_error',optimizer='adam',metrics=['accuracy'])
model.fit(x_train,y_train,epochs=15,validation_split=0.2)

ypred = (model.predict(x_test) > 0.5).astype('int32')
print("Accuracy: ",accuracy_score(ypred,y_test)*100)