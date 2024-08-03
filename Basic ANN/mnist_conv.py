#accuracy = 99.7
import mnist
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense,Conv2D,MaxPooling2D,Flatten,BatchNormalization,Dropout
from sklearn.metrics import accuracy_score

(X_train,y_train),(X_test,y_test) = keras.datasets.mnist.load_data()
X_train = X_train/255
X_test = X_test/255


model = Sequential()

model.add(Conv2D(32,kernel_size=(3,3),activation='relu',input_shape=(66,66,1)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2),strides=2,padding='valid'))

model.add(Conv2D(64,kernel_size=(3,3),padding='valid',activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2),strides=2,padding='valid'))

model.add(Conv2D(128,kernel_size=(3,3),padding='valid',activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2),strides=2,padding='valid'))


 
model.add(Flatten())
model.add(Dense(1024,activation='relu'))
model.add(Dropout(0.05))
model.add(Dense(512,activation='relu'))
model.add(Dropout(0.05))
model.add(Dense(256,activation='relu'))
model.add(Dropout(0.05))
model.add(Dense(128,activation='relu'))
model.add(Dropout(0.05))
model.add(Dense(64,activation='relu'))
model.add(Dropout(0.05))
model.add(Dense(10,activation='softmax'))


print(model.summary())

#model ready
model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

model.fit(X_train,y_train,
            epochs=1,
            validation_split=0.1)

print("TESTING FROM THE GIVEN DATASET: ")

y_prob = model.predict(X_test)
y_pred = model.predict(X_test).agmax()

for i in range(5):
    prediction = model.predict(X_test[i].reshape(1, 28, 28, 1))
    print("Predicted value of local dataset:", prediction.argmax())
    print("Actual value: ",y_test[i])
    plt.imshow(X_test[i])    
    plt.show()


# #custom test case
# print("Testing from Local Test case: ")
# test = pd.read_csv('C:/Users/EIOT/Desktop/result.csv')

# # Drop the 'value' column to get the pixel values
# X_test = test.drop(columns='Number').values
# y_test=test['Number']
# # X_test = abs(1-( X_test/255))
# X_test=X_test/225
# X_test = X_test.reshape((-1,28, 28,1))

# y_prob = model.predict(X_test)
# y_pred = model.predict(X_test).argmax(axis=1)
# print("accuracy: ",accuracy_score(y_test,y_pred)*100)

# for i in range(len(X_test)):
#     prediction = model.predict(X_test[i].reshape(1, 28, 28, 1))
#     print("Predicted value of local dataset:", prediction.argmax())
#     print("Actual value: ",y_test[i])
#     plt.imshow(X_test[i])    
#     plt.show()


