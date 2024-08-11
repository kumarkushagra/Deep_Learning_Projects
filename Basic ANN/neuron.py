#80% accuracy

import pandas as pd
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
 
data = pd.read_csv('D:/NN/New folder (2)/heart.csv')
x=data.drop(columns='target')
y=data['target']

print(data.shape)
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.1,random_state=1)



scaler = MinMaxScaler()
x_train  = scaler.fit_transform(x_train)
x_test  = scaler.transform(x_test)

model = Sequential()
model.add(Dense(500, activation='relu', input_shape=(11,)))
model.add(Dense(700, activation='relu'))
model.add(Dense(900, activation='relu'))
model.add(Dense(500, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

print(model.summary())

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(x_train,y_train,epochs=10,validation_split=0.01)

y_prob = model.predict(x_test)
y_pred = (y_prob > 0.5).astype('int32')
print("accuracy: ",accuracy_score(y_test,y_pred)*100)

