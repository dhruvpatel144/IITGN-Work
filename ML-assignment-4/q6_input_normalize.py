# -*- coding: utf-8 -*-
"""Q6_input_normalize.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1f7kCUuDOHfftApIrWG1t-OUn2FVdaxeV
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from linearRegression.linear_regression import LinearRegression
import pandas as pd
from metrics import *
#TODO : Write here

np.random.seed(45)

N=60
x = np.array([i*np.pi/180 for i in range(60,300,2)])
np.random.seed(10)
y = 3*x + 8 + np.random.normal(0,3,len(x)) 

X = x.reshape(-1,1)
one_mat = np.ones((X.shape[0],1)) 
X = np.hstack((one_mat,X)) 


y=pd.Series(y)
LR = LinearRegression(fit_intercept=True)
LR.fit_gradient_descent(pd.DataFrame(X), y, batch_size=60,gradient_type='jax',penalty_type='l2',num_iters=50,lr=0.01)

a1= LR.all_coef[0]  
a2= LR.all_coef[1] 

LR.plot_line_fit(X[:, 1], y, a1[49], a2[49])




N=60
x = np.array([i*np.pi/180 for i in range(60,300,2)])
np.random.seed(10)
y = 3*x + 8 + np.random.normal(0,3,len(x)) 

sc1 = StandardScaler()
x = sc1.fit_transform(x.reshape((-1, 1)))
sc2 = StandardScaler()
y = sc2.fit_transform(y.reshape((-1, 1))).flatten()

X = x.reshape(-1,1)
one_mat = np.ones((X.shape[0],1)) 
X = np.hstack((one_mat,X)) 


y=pd.Series(y)
LR = LinearRegression(fit_intercept=True)
LR.fit_gradient_descent(pd.DataFrame(X), y, batch_size=60,gradient_type='jax',penalty_type='l2',num_iters=50,lr=0.01)

a1= LR.all_coef[0]  
a2= LR.all_coef[1] 

LR.plot_line_fit(X[:, 1], y, a1[49], a2[49])
