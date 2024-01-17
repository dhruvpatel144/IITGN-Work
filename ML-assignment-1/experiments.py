
from matplotlib.font_manager import FontProperties
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import *
from mpl_toolkits import mplot3d

np.random.seed(42)
num_average_time = 100
x_plt = [j for j in range(10, 50, 10) for _ in range(2,5)]
y_plt = [_ for j in range(10, 50, 10) for _ in range(2,5)]

# Learn DTs 
# ...
# 
# Function to calculate average time (and std) taken by fit() and predict() for different N and P for 4 different cases of DTs
# ...
# Function to plot the results
# ..
# Function to create fake data (take inspiration from usage.py)
# ...
# ..other functions

def random_data(m,n,i):
    if i==1:
        X = pd.DataFrame(np.random.randn(n, m))
        y = pd.Series(np.random.randn(n))
    elif i==2: 
        X = pd.DataFrame(np.random.randn(n, m))
        y = pd.Series(np.random.randint(m, size = n), dtype="category")
    elif i==3: 
        X = pd.DataFrame({i:pd.Series(np.random.randint(m, size = n), dtype="category") for i in range(5)})
        y = pd.Series(np.random.randn(n))
    else: 
        X = pd.DataFrame({i:pd.Series(np.random.randint(m, size = n), dtype="category") for i in range(5)})
        y = pd.Series(np.random.randint(m, size = n),  dtype="category")
        
    return X, y


fit_time = []
predict_time=[]
samples=[]
#Case 1: Real input Real Output

for m in range(2,5):
    for n in range(10,50,10):
        X,y = random_data(m,n,1)
        
        clf = DecisionTree(criterion="information_gain")
        start_time = time.time()
        clf.fit(X,y)
        end_time = time.time()

        fit_time.append((end_time-start_time)*1000)

        start_time_p = time.time()
        y_pred = clf.predict(X)
        end_time_p = time.time()

        predict_time.append((end_time_p-start_time_p)*1000)
        samples.append(n)

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_trisurf(x_plt, y_plt, fit_time, cmap='viridis', linewidth=0.5)
plt.title("Time for Learning, RIRO type")
plt.xlabel("N")
plt.ylabel("M")
plt.show()
plt.plot(list(fit_time))
plt.xlabel("Total Fit time Values")
plt.ylabel("Real Input Real Output fit/training time", fontsize=12)
plt.savefig("RIRO_train.pdf")
plt.show()

plt.plot(list(predict_time))
plt.xlabel("Total Predict time Values")
plt.ylabel('"Real Input Real Output predict/inference time', fontsize=12)
plt.savefig("RIRO_test.png")
plt.show()

# x_plt = [j for j in range(10,40, 10) for _ in range(2,5) ]
# y_plt = [_ for j in range(10,40, 10) for _ in range(2,5) ]

#Case 2: Real Input Discrete Output 
fit_time_2 = []
predict_time_2=[]
samples_2=[]
for m in range(2,5):
    for n in range(10,50,10):

        X,y = random_data(m,n,2)
        
        clf = DecisionTree(criterion="information_gain")
        start_time = time.time()
        clf.fit(X,y)
        end_time = time.time()

        fit_time_2.append((end_time-start_time)*1000)

        start_time_p = time.time()
        y_pred = clf.predict(X)
        end_time_p = time.time()
        print(m,n)
        predict_time_2.append((end_time_p-start_time_p)*1000)
        samples_2.append(n)

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_trisurf(x_plt, y_plt, fit_time_2, cmap='viridis', linewidth=0.5)
plt.title("Time for Learning, RIDO type")
plt.xlabel("N")
plt.ylabel("M")
plt.show()


plt.plot(list(fit_time_2))
plt.xlabel("Total Fit time Values")
plt.ylabel("Real Input Discrete Output fit/training time", fontsize=12)
plt.savefig("RIDO_train.png")
plt.show()

plt.plot(list(predict_time_2))
plt.xlabel("Total Predict time Values")
plt.ylabel('Real Input Discrete Output: Predict time', fontsize=12)
plt.savefig("RIDO_test.png")
plt.show()

#Case 3: Discrete input Real Output

fit_time_3 = []
predict_time_3=[]
samples_3=[]
for m in range(2,5):
    for n in range(10,50,10):
        X,y = random_data(m,n,3)
        
        clf = DecisionTree(criterion="gini_index")
        start_time = time.time()
        clf.fit(X,y)
        end_time = time.time()

        fit_time_3.append((end_time-start_time)*1000)

        start_time_p = time.time()
        y_pred = clf.predict(X)
        end_time_p = time.time()
        samples_3.append(n)
        predict_time_3.append((end_time_p-start_time_p)*1000)

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_trisurf(x_plt, y_plt, fit_time_3, cmap='viridis', linewidth=0.5)
plt.title("Time for Learning, DIRO type")
plt.xlabel("N")
plt.ylabel("M")
plt.show()

plt.plot(list(fit_time_3))
plt.xlabel("Total Fit time Values")
plt.ylabel("Discrete Input Real Output fit/training time", fontsize=12)
plt.savefig("DIRO_train.png")
plt.show()

plt.plot(list(predict_time_3))
plt.xlabel("Total Predict time Values")
plt.ylabel('Discrete Input Real Output predict/inference time', fontsize=12)
plt.savefig("DIRO_test.png")
plt.show()

#Case 4: Discrete input Discrete Output

fit_time_4 = []
predict_time_4=[]
samples_4=[]
for m in range(2,5):
    for n in range(10,50,10):
        X,y = random_data(m,n,4)
        
        clf = DecisionTree(criterion="gini_index")
        start_time = time.time()
        clf.fit(X,y)
        end_time = time.time()

        fit_time_4.append((end_time-start_time)*1000)

        start_time_p = time.time()
        y_pred = clf.predict(X)
        end_time_p = time.time()
        samples_4.append(n)
        predict_time_4.append((end_time_p-start_time_p)*1000)

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_trisurf(x_plt, y_plt, fit_time_4, cmap='viridis', linewidth=0.5)
plt.title("Time for Learning, DIDO type")
plt.xlabel("N")
plt.ylabel("M")
plt.show()
        
plt.plot(list(fit_time_4))
plt.xlabel("Total Fit time Values")
plt.ylabel("Discrete Input Discrete Output fit/training time", fontsize=12)
plt.savefig("DIDO_train.png")
plt.show()

plt.plot(list(predict_time_4))
plt.xlabel("Total Predict time Values")
plt.ylabel('"Discrete Input Discrete Output predict/test time', fontsize=10)
plt.savefig("DIDO_test.png")
plt.show()


