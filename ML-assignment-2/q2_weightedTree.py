import pandas as pd
import numpy as np
from tree.base import DecisionTree
from metrics import *
import random
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification
X, y = make_classification(n_features=2, n_redundant=0, n_informative=2, random_state=1, n_clusters_per_class=2, class_sep=0.5)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=42)

df_Xtrain = pd.DataFrame(X_train)
df_Xtest = pd.DataFrame(X_test)
df_ytrain = pd.Series(y_train)
df_ytest = pd.Series(y_test)

df_X = pd.DataFrame(X)
df_y = pd.Series(y)
df_y_cat = df_y.astype('category')

weights = np.random.uniform(0, 1, len(y_train))
weights = pd.Series(weights)
# Weighted Decision Tree
# Training the Weighted Decision Tree
my_tree = DecisionTree(criterion = "gini")
my_tree.fit(df_Xtrain, df_ytrain,weights)
# plotting the tree    
# making predictions from the weighted decision tree
my_pred = my_tree.predict(df_Xtest)  
plotty = my_tree.plot(df_X,df_y_cat,my_tree)
my_acc = accuracy(my_pred, y_test)   

# sklearn's decision tree
skl_tree = DecisionTreeClassifier(criterion="gini")
skl_tree.fit(df_Xtrain, df_ytrain, sample_weight=weights)
skl_pred = skl_tree.predict(df_Xtest)
skl_acc = accuracy(skl_pred, y_test) 


print(f"Accuracy of the weighted decision tree = {my_acc:.2f}")
print(f"Accuracy of decision tree from sklearn = {skl_acc}")

