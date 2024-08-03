from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense,Flatten
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

(X_train,y_train),(X_test,y_test) = keras.datasets.mnist.load_data()
print(X_train.shape)
X_train = X_train/255
X_test = X_test/255

#creating the basic archetecture of the model

model = Sequential()
model.add(Flatten(input_shape=(28,28)))
model.add(Dense(128,activation='relu'))
model.add(Dense(32,activation='relu'))
model.add(Dense(10,activation='softmax'))

print(model.summary())

model.compile(loss='sparse_categorical_crossentropy',
            optimizer='Adam')

model.fit(X_train,y_train,
                epochs=10,
                validation_split=0.2)

y_prob = model.predict(X_test)
y_pred = y_prob.argmax(axis=1)
print("accuracy; ",accuracy_score(y_test,y_pred)*100)


for i in range(27,30):
    plt.imshow(X_test[i])    
    plt.show()
    print("Predicted value: ",model.predict(X_test[i].reshape(1,28,28)).argmax(axis=1))