from typing import Union
import pandas as pd
import numpy as np


def accuracy(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the accuracy
    """

    """
    The following assert checks if sizes of y_hat and y are equal.
    Students are required to add appropriate assert checks at places to
    ensure that the function does not fail in corner cases.
    """
    assert(y_hat.size == y.size)
    if isinstance(y,pd.Series):
        y= y.values
    return float((sum(y_hat == y) / (len(y_hat))))
    


def precision(y_hat: pd.Series, y: pd.Series, cls: Union[int, str]) -> float:
    """
    Function to calculate the precision
    """
    assert(y_hat.size == y.size)
    if isinstance(y,pd.Series):
        y= y.values
    true_pos=0.0
    false_pos=0.0
    for i in range(len(y)):
        if(y_hat[i]==cls):
            if(y[i]==cls):
                true_pos+=1
            elif(y[i]!=cls):
                false_pos+=1
    return (true_pos/float(true_pos+false_pos))
    


def recall(y_hat: pd.Series, y: pd.Series, cls: Union[int, str]) -> float:
    """
    Function to calculate the recall
    """
    assert(y_hat.size == y.size)
    if isinstance(y,pd.Series):
        y= y.values
    true_pos=0.0
    false_neg=0.0
    for i in range(len(y_hat)):
        if(y[i]==cls):
            if(y_hat[i]==cls):
                true_pos+=1
            elif(y_hat[i]!=cls):
                false_neg+=1
    return (true_pos/float(true_pos+false_neg))



def rmse(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the root-mean-squared-error(rmse)
    """
    assert y_hat.size == y.size
    ans = np.sqrt(((y-y_hat)**2).mean())
    return ans


def mae(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the mean-absolute-error(mae)
    """
    assert y_hat.size == y.size
    assert y.size>0
    ans = abs(y-y_hat).sum()/(y.size)
    return ans
