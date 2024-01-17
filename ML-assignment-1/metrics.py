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
    assert y_hat.size == y.size
    # TODO: Write here
    ans = (y_hat==y).sum()/y.size
    return ans
    


def precision(y_hat: pd.Series, y: pd.Series, cls: Union[int, str]) -> float:
    """
    Function to calculate the precision
    """
    assert y_hat.size == y.size
    if(y_hat==cls).sum()>0:
        numerator =0
        for i in range(len(y)):
            if((y[i]==cls) and (y_hat[i]==cls)):
                numerator+=1

        ans = numerator/((y_hat==cls).sum())
    else:
        return None
    return ans
    


def recall(y_hat: pd.Series, y: pd.Series, cls: Union[int, str]) -> float:
    """
    Function to calculate the recall
    """
    assert y_hat.size == y.size
    if(y==cls).sum()>0:
        num=0
        for i in range(len(y)):
            if((y[i]==cls) and (y_hat[i]==cls)):
                num+=1
        ans = num/((y==cls).sum())
    else:
        return None
    return ans



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
