import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import *
from sklearn.model_selection import train_test_split, KFold, cross_val_score
import pandas as pd


np.random.seed(42)

from sklearn.datasets import make_classification
X, y = make_classification(
n_features=2, n_redundant=0, n_informative=2, random_state=1, n_clusters_per_class=2, class_sep=0.5)

# For plotting
import matplotlib.pyplot as plt
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.show()

train_x, test_x, train_y, test_y = train_test_split(X,y, train_size = 0.7, random_state=42)

# print(train_x.shape, train_y.shape)


X_train = pd.DataFrame(train_x)
y_train = pd.Series(train_y, dtype = y.dtype)
X_test = pd.DataFrame(test_x)
y_test = pd.Series(test_y, dtype = y.dtype)

classification= DecisionTree(criterion='information_gain',max_depth=5)
classification.fit(X_train,y_train)
y_pred = classification.predict(X_test)

print("Accuracy on the test dataset: ", accuracy(y_pred,y_test))
for classs in list(set(y)):
    print("Precision for class " + str(classs)+ " is ", precision(y_pred,y_test,classs))
    print("Recall for class " + str(classs)+ " is ", recall(y_pred,y_test,classs))


cross_val= KFold(n_splits=5, random_state=1, shuffle=True)
opti_values= {0:dict(),1:dict(),2:dict(),3:dict(),4:dict()}

for fold, (train_idx,test_idx) in (enumerate(cross_val.split(X=X,y=y))):
    X_train, X_validation= pd.DataFrame(X[train_idx]), pd.DataFrame(X[test_idx])
    y_train,y_validation = y[train_idx], y[test_idx]
    max_acc=0
    optimum_depth=2
    for k in range(2,8):
        classification = DecisionTree(criterion='gini-index',max_depth=k)
        classification.fit(X_train,y_train)
        y_pred = classification.predict(X_validation)
        acc= accuracy(y_pred,y_validation)
        if(acc>max_acc):
            optimum_depth=k
            max_acc= acc
    opti_values[fold][optimum_depth]= max_acc
    
    
opt_acc=0
opt_depth=0

for key, value in opti_values.items():
    for key1, val in value.items():
        print("optimal depth for fold " + str(key) +" is " + str(key1)+" and Accuracy is "+str(val))
for key,value in opti_values.items():
    for key1,val in value.items():
        if(val>opt_acc):
            opt_acc = val
            opt_depth = key1
print("Best Depth is ", opt_depth)
print("Best Accuracy is ", opt_acc)





# Read dataset
# ...
# 
