import pandas as pd
import numpy as np
import math


def entropy(Y: pd.Series, weights: pd.Series = None) -> float:
    """
    Function to calculate the entropy of a Pandas Series with weighted samples
    """
    y_list = list(Y)
    y_size = len(y_list)

    if weights is not None:                # weighted samples
        weights = list(weights)
        weights_sum = sum(weights)      # sum of all weights
        counts = {}                     # to store the frequency of each class label in Y
        for i, j in enumerate(y_list):
            if j not in counts:
                counts[j] = 0
            counts[j] += weights[i]     # the weighted frequency is added 

        # calculating the entropy
        entropy = 0
        for i in counts: 
            p = counts[i] / weights_sum     # weighted probability
            if p > 0:
                entropy -= p * math.log2(p)

    else:                     # samples not weighted
        count = {}          # to store the count of each attribute in the column

        for i in y_list:
            if i in count:
                count[i] += 1          # incrementing the count of each attribute when they repeat
            else:
                count[i] = 1

        entropy = 0
        for i in count:
            p = count[i] / y_size        # calcualting the probability of each attribute
            entropy -= p * math.log2(p)  # implementing the formula H(X) = -summation(p(x_i) * log2(p(x_i)))

    return entropy
    pass


def gini_index(Y: pd.Series) -> float:
    """
    Function to calculate the gini index
    """
    count = {}
    y_list = list(Y)
    y_size = Y.size

    for i in y_list:
        if i in count:
            count[i] += 1
        else:
            count[i] = 1 

    gini_ind = 1
    for i in count:
        p = count[i] / y_size
        gini_ind -= p**2

    return gini_ind 
    pass


def information_gain(Y: pd.Series, attr: pd.Series, weights=None) -> float:
    """
    Function to calculate the information gain
    """
    count = {}            # to store the occurences of each unique element in this particular attribute column
    y_list = list(Y)
    y_size = len(y_list)
    attr_list = list(attr)

    if weights is not None:           # weighted samples
        weights = list(weights)
        weights_sum = sum(weights)
        
        for i in range(attr.size):
            if attr_list[i] in count:
                count[attr_list[i]].append((y_list[i], weights[i]))
            else:
                count[attr_list[i]] = [(y_list[i], weights[i])]        # add tuples with the value of y corresponding to the data point along with the weight of the data point
                
        info_gain = entropy(y_list, weights)        # initialising with entropy from with further specific attrubute entropies are to be subtracted
        
        for i in count:
            i_counts = [j[0] for j in count[i]]             
            i_weights = [j[1] for j in count[i]]
            info_gain -= (sum(i_weights) / weights_sum) * entropy(i_counts, i_weights)           # weighted info gain
            
    else:                          # samples not weighted
        for i in range(attr.size):             # iterating attr_list to access each element of the column
            if attr_list[i] in count:
                count[attr_list[i]].append(y_list[i])  
            else:
                count[attr_list[i]] = [y_list[i]]

        info_gain = entropy(y_list)
        
        for i in count:
            info_gain -= (len(count[i]) / y_size) * entropy(count[i])      # implementing the formula Gain = Entropy(S) - summation ((|S_v| / |S|) * entropy(S_v))

    return info_gain

