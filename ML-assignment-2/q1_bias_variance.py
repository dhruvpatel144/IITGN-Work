import numpy as np
import sklearn
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os
# from tree.base import DecisionTree

np.random.seed(1234)
x = np.linspace(0, 10, 50)
eps = np.random.normal(0, 5, 50)
# print(eps)
y = x**2 + 1 + eps
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=42)
x_train = x_train.reshape(-1,1)
x_test = x_test.reshape(-1,1)

number_of_trees = 100
def bias(x_train,y_train, depth , number_of_trees):
    y_hat_array = []
    for i in range(number_of_trees):
        np.random.seed(i)
        X_train,_, Y_train,_ = train_test_split(x_train, y_train, train_size=0.7, random_state=i)
        X_train = X_train.reshape(-1,1)
        Y_train = Y_train.reshape(-1,1)
        model = DecisionTreeRegressor(max_depth = depth)
        model.fit(X_train, Y_train)
        y_hat = model.predict(x_train)
    y_hat_array.append(y_hat)
    difference = np.mean(abs(np.mean(y_hat_array, axis = 0) - y_train))
    return difference

def variance(x_train, x_test, y_train, depth, number_of_trees):
    y_hat_array = []
    for i in range(number_of_trees):
        np.random.seed(i)
        X_train,_, Y_train,_ = train_test_split(x_train, y_train, train_size=0.7, random_state=i)
        X_train = X_train.reshape(-1,1)
        Y_train = Y_train.reshape(-1,1)
        model = DecisionTreeRegressor(max_depth = depth)
        model.fit(X_train, Y_train)
        y_hat = model.predict(x_test)
        y_hat_array.append(y_hat)
    variances = np.var(y_hat_array, axis = 1)
    return np.mean(variances)
depths = np.arange(1,11)
bias_array = []
variance_array = []
for depth in depths:
    bias_array.append(bias(x_train,y_train,depth, number_of_trees))
    variance_array.append(variance(x_train,x_test, y_train,depth,number_of_trees))
bias_array = np.array(bias_array)
variance_array = np.array(variance_array)
normalized_bias_array = (bias_array-bias_array.min())/(bias_array.max()-bias_array.min())
normalized_variance_array = (variance_array-variance_array.min())/(variance_array.max()-variance_array.min())
plt.plot(depths, normalized_bias_array)
plt.plot(depths, normalized_variance_array)
plt.legend(['bias', 'variance'])
plt.savefig(os.path.join("figures", "Q1_Fig1.png"))
plt.show()



# plt.plot(x, y, 'o')
# plt.plot(x, x**2 + 1, 'r-')
# plt.show()
