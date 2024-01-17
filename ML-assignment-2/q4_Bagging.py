import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from metrics import *
import time

from ensemble.bagging import BaggingClassifier
from tree.base import DecisionTree
from sklearn.tree import DecisionTreeClassifier

# Or use sklearn decision tree

########### BaggingClassifier ###################

N = 30
P = 2
NUM_OP_CLASSES = 2
n_estimators = 3
X = pd.DataFrame(np.abs(np.random.randn(N, P)))
y = pd.Series(np.random.randint(NUM_OP_CLASSES, size=N), dtype="category")

criteria = 'information_gain'
tree = DecisionTreeClassifier
Classifier_B = BaggingClassifier(
    base_estimator=tree, n_estimators=n_estimators,
    criterion="gini", n_jobs = n_estimators)
Classifier_B.fit(X, y)
y_hat = Classifier_B.predict(X)
[fig1, fig2] = Classifier_B.plot(X, y)
print('Criteria :', criteria)
print('Accuracy: ', accuracy(y_hat, y))
for cls in y.unique():
    print('Precision: ', precision(y_hat, y, cls))
    print('Recall: ', recall(y_hat, y, cls))

print("Plots saved as Q6_Fig1.png and Q6_Fig2.png")
parallel_time = []
non_parallel_time = []
for N in [30, 100, 500, 1000, 5000, 10000]:
    P = 2
    NUM_OP_CLASSES = 2
    n_estimators = 3
    start1 = time.time()
    X = pd.DataFrame(np.abs(np.random.randn(N, P)))
    y = pd.Series(np.random.randint(NUM_OP_CLASSES, size=N), dtype="category")

    criteria = 'information_gain'
    tree = DecisionTreeClassifier
    Classifier_B = BaggingClassifier(
        base_estimator=tree, n_estimators=n_estimators,
        criterion="gini", n_jobs = n_estimators)
    Classifier_B.fit(X, y)
    y_hat = Classifier_B.predict(X)
    end1 = time.time()
    time_taken = (end1-start1)*(10)**3
    parallel_time.append(time_taken)

for N in [30, 100, 500, 1000, 5000, 10000]:
    P = 2
    NUM_OP_CLASSES = 2
    n_estimators = 3
    start = time.time()
    X = pd.DataFrame(np.abs(np.random.randn(N, P)))
    y = pd.Series(np.random.randint(NUM_OP_CLASSES, size=N), dtype="category")

    criteria = 'information_gain'
    tree = DecisionTreeClassifier
    Classifier_B = BaggingClassifier(
        base_estimator=tree, n_estimators=n_estimators,
        criterion="gini", n_jobs = 1)
    Classifier_B.fit(X, y)
    y_hat = Classifier_B.predict(X)
    end = time.time()
    time_taken = (end-start)*(10)**3
    non_parallel_time.append(time_taken)
print(parallel_time)
print(non_parallel_time)


