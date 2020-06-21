import pandas as pd
from pydataset import data
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

pima = data("Pima.tr")
print(pima)
pima.plot(kind="scatter", x="skin", y="bmi")
x_train, x_test, y_train, y_test = train_test_split(pima.skin, pima.bmi)

#plt.scatter(x_train, y_train, label="Training Data", color='r')
#plt.scatter(x_test, y_test, label="Testing Data", color='b')
#plt.legend()


lr = LinearRegression()
lr.fit(x_train.values.reshape(-1,1), y_train)

y_predicted = lr.predict(x_test.values.reshape(-1,1))

plt.plot(x_test, y_predicted, color='r')
plt.scatter(x_test, y_test, color= 'b')
plt.show()

a = np.array([50])
print(a.ndim)
print(lr.predict(a.reshape(-1,1)))

