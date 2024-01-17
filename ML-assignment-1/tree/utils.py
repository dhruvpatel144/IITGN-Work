import pandas as pd
import numpy as np
import math


def entropy(Y: pd.Series) -> float:
    """
    Function to calculate the entropy
    """
    # probs = Y.value_counts(normalize = True)
    # ent = -((probs)*(np.log2(probs))).sum()
    # return ent
    entropy = 0
    unique_values = np.bincount(Y) # storing the count of all unique values present in Y.
    probabilities = unique_values / len(Y) #probability of each class present inside Y
    for prob in probabilities:
        if prob > 0: 
            entropy += prob*math.log(prob,2)
    return -entropy


def gini_index(Y: pd.Series) -> float:
    """
    Function to calculate the gini index
    """
    # probs = Y.value_counts(normalize = True)
    # gini = (probs*(1-probs)).sum()
    # return gini

    gini = 0
    unique_values = np.bincount(Y) 
    probabilities = unique_values / len(Y) #probability of each class present in Y
    for prob in probabilities:
        gini+= prob**2
    return 1-gini


def information_gain(Y: pd.Series, attr: pd.Series) -> float:
    """
    Function to calculate the information gain
    """
    assert(Y.size == attr.size)
    assert(Y.size > 0)
    d= {}
    for i in range(len(attr)):
        if attr[i] in d:
            d[attr[i]].append(Y[i])
        else:
            d[attr[i]] = [Y[i]]
    inform_gain=0
    for i in d:
        inform_gain += (len(d[i])/len(Y))*entropy(d[i])
    inform_gain= entropy(Y)-inform_gain
    return inform_gain

    
