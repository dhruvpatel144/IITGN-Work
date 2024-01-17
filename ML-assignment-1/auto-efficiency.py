
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import *
from sklearn.tree import DecisionTreeRegressor

np.random.seed(42)

data= pd.read_csv("auto-mpg.csv")
data.drop("car name",axis=1,inplace=True)
data['horsepower'] = pd.to_numeric(data['horsepower'],errors = 'coerce') #converting the horsepower from object type to float type
data["horsepower"]= data["horsepower"].fillna(data["horsepower"].mean()) #adding zeros where nan is present
data = data.astype("float") #convert all fields to float
training_data = data.sample(frac=0.7, random_state=42) #train-test split into 70% training and 30% testing
testing_data = data.drop(training_data.index)  #train-test split into 70% training and 30% testing

X_train= training_data.iloc[:,1:]  
y_train=training_data.iloc[:,0].values
X_test= testing_data.iloc[:,1:]
y_test=testing_data.iloc[:,0].values

X_train= pd.DataFrame(X_train.values)
X_test = pd.DataFrame(X_test.values)
y_train= pd.Series(y_train)
y_test = pd.Series(y_test)

print(X_train)


our_classfication= DecisionTree(criterion='information_gain',max_depth=3)
our_classfication.fit(X_train,y_train)
y_pred1 = our_classfication.predict(X_test)
print("RMSE of our decision tree ", rmse(y_pred1,y_test))
print("MAE of our decision tree ", mae(y_pred1,y_test))

# Inbuilt sklearn
sklearn_classification = DecisionTreeRegressor(max_depth=3)
sklearn_classification.fit(X_train, y_train)
y_pred = sklearn_classification.predict(X_test)
print("RMSE of sklearn decision tree ", rmse(y_pred,y_test))
print("MAE of sklearn decision tree ", mae(y_pred,y_test))




# Read real-estate data set
# ...
# 
