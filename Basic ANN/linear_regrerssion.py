import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plott(X, y, Z):
    plt.scatter(X, y)
    plt.plot(X, Z,color='red')
    plt.show()

def layer1(X, W, b):
    return np.dot(X, W) + b

def differential(X, y, W, b):
    yhat = layer1(X, W, b)
    dW = np.dot(X.T, (yhat - y)) / len(X)
    db = np.mean(yhat - y)
    return dW, db

def updating(X, y, W, b, eta, epochs):
    for _ in range(epochs):
        dW, db = differential(X, y, W, b)
        W -= eta * dW
        b -= eta * db
        Z = layer1(X, W, b)
        plott(areas, prices, Z)
        print(W,b)
        
data = pd.read_csv("D:/Development/Dataset/housing price/data.csv")
x = data["sqft_lot"]
y = data["price"]

areas = np.array(x)
prices = np.array(y)

areas=areas/areas.mean()
prices = prices/prices.max()

updating(areas, prices, 10, 10, 0.1, 150)
