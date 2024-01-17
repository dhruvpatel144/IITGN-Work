import numpy as np

def rmse(y_true, y_pred):
    """
    Calculate the root mean squared error between two arrays.
    """
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def mae(y_true, y_pred):
    """
    Calculate the mean absolute error between two arrays.
    """
    return np.mean(np.abs(y_true - y_pred))