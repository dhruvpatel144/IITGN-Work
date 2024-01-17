# -*- coding: utf-8 -*-
"""Q7_forecast.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1QygJyzTuf4wJ3ioxCqF0boFIbPNbBJd6
"""

#TODO : Write here
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

data = pd.read_csv('https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv', header=0, index_col=0, parse_dates=True)

train_size = int(len(data) * 0.8)
train_data, test_data = data[:train_size], data[train_size:]

lag = 3  # number of lagged temperature values to use as predictors
model = LinearRegression()

X_train = train_data['Temp'].values[:-lag]
y_train = train_data['Temp'].values[lag:]
X_test = test_data['Temp'].values[:-lag]
y_test = test_data['Temp'].values[lag:]


X_train = X_train.reshape(-1, 1)
y_train = y_train.reshape(-1, 1)
model.fit(X_train, y_train)


X_test = X_test.reshape(-1, 1)
y_test_pred = model.predict(X_test)


plt.scatter(test_data.index[lag:], y_test, label='True')
plt.plot(test_data.index[lag:], y_test_pred, label='Predicted', color='orange')
plt.xticks(rotation=30)
plt.legend()
plt.show()


rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
print('RMSE:', rmse)