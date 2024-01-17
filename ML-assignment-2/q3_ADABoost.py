import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from metrics import *

from ensemble.ADABoost import AdaBoostClassifier
from tree.base import DecisionTree
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier

# Or you could import sklearn DecisionTree

np.random.seed(42)

########### AdaBoostClassifier on Real Input and Discrete Output ###################


N = 30
P = 2
NUM_OP_CLASSES = 2
n_estimators = 3
X = pd.DataFrame(np.abs(np.random.randn(N, P)))
y = pd.Series(np.random.randint(NUM_OP_CLASSES, size=N), dtype="category")


criteria = "entropy"
tree = DecisionTreeClassifier(criterion=criteria)
Classifier_AB = AdaBoostClassifier(base_estimator=tree, n_estimators=n_estimators)
Classifier_AB.fit(X, y)
y_hat = Classifier_AB.predict(X)
[fig1, fig2] = Classifier_AB.plot(X,y)
print("Criteria :", criteria)
print("Accuracy: ", accuracy(y_hat, y))
for cls in y.unique():
    print("Precision: ", precision(y_hat, y, cls))
    print("Recall: ", recall(y_hat, y, cls))

# -------------------------------------------Classification Dataset------------------------------------------------

X, y = make_classification(n_samples=30,n_features=2, n_redundant=0, n_informative=2, random_state=15, n_clusters_per_class=2, class_sep=0.5, n_classes=2)

# # For plotting
import matplotlib.pyplot as plt
plt.scatter(X[:, 0], X[:, 1], c=y)

X=pd.DataFrame(X)
X['y']=y

data_f= X.sample(frac=1, random_state=0)
data_f.reset_index(drop=True, inplace=True)
y= data_f.pop('y')

split_range= int(0.6*len(X))
X_train= data_f[:split_range]
X_test = data_f[split_range:]
y_train= pd.Series(y[:split_range],dtype=y.dtype)
y_test = pd.Series(y[split_range:],dtype=y.dtype)

n_estimators=3
tree= DecisionTreeClassifier(criterion='entropy',max_depth=1)
tree.fit(X_train, y_train)
y_pred_stump = pd.Series(tree.predict(X_test))
clf = AdaBoostClassifier(base_estimator=tree, n_estimators= n_estimators)
clf.fit(X_train, y_train)
y_pred = pd.Series(clf.predict(X_test))

clf.plot(X,y)

print("Accuracy from Adaboost:", accuracy(y_pred,y_test))
print("Accuracy from stump:", accuracy(y_pred_stump,y_test))

for cls in y.unique():
    print('Precision for',cls,'in Adaboost : ', precision(y_pred, y_test, cls))
    print('Precision for',cls,'in stump : ', precision(y_pred_stump, y_test, cls))
    print('Recall for ',cls ,'in Adaboost: ', recall(y_pred, y_test, cls))
    print('Recall for ',cls ,'in stump: ', recall(y_pred_stump, y_test, cls))