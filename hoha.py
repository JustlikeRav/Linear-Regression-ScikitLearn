import csv
import matplotlib.pyplot as plt
import pandas
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

dataset = pandas.read_csv("data.csv")
Prices = dataset.iloc[:, 0].values
Area = dataset.iloc[:, 1].values

Area = np.reshape(Area, (Area.size, 1))
Prices = np.reshape(Prices, (Area.size, 1))

xTrain, xTest, yTrain, yTest = train_test_split(Area, Prices, test_size=1 / 3, random_state=0)

linearRegressor = LinearRegression()
linearRegressor.fit(xTrain, yTrain)

# yPrediction = linearRegressor.predict(xTest)

plt.scatter(xTrain, yTrain, s=30, color="#777777")
plt.title('Prices vs Area')
plt.xlabel('Area')
plt.ylabel('Price')
plt.plot(xTest, linearRegressor.predict(xTest), color="green")
plt.scatter(xTest, yTest, s=20, color="red")
plt.show()

# print(Prices)
