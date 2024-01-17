"""
The current code given is for the Assignment 1.
You will be expected to use this to make trees for:
> discrete input, discrete output
> real input, real output
> real input, discrete output
> discrete input, real output
"""
from dataclasses import dataclass
from typing import Literal, Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tree.utils import entropy, information_gain, gini_index

np.random.seed(42)


@dataclass

class Node():
    def __init__(self):
        self.val = None # value of the parameter is stored.
        self.isleaf = False #True if node is a leaf else false
        self.attr_number_idx = None #for storing the index of the feature to be split
        self.splitvalue = None # the value at which the split occurs/threshold
        self.isCategorical= False # True in case of categorical data otherwise false.
        self.tree_child = {} # Dict to store children 

# for real output

class DecisionTreeRegressor(): 
    
    def __init__(self, criteria, max_depth=None):
        self.critera = criteria
        self.max_depth = max_depth
        self.head = None
        
    def split_fit(self,X,y,depth):
        current_node = Node()   # node created
        current_node.attr_number_idx = -1
        split_value = None # used to store the value of the split
        criteria_value = None # used to store final criteria value      
          
        classes=np.unique(y) 
        
        # if no features are present or we have exhausted the max depth
        if((X.shape[1]==0) or ((self.max_depth!=None) and (depth==self.max_depth))):
            current_node.isleaf = True
            current_node.val = y.mean()  # we return the mean of all the y in that class.
            return current_node

        # if only one class is present
        if(len(classes)==1): 
            current_node.isleaf = True
            current_node.val = classes[0] # we assign that class
            return current_node

        for feature in X:
            x = X[feature]

            # We check whether the ouput is discrete or not by checking the dtype of the variable.
            if(x.dtype.name=="category"):
                classes_unique = np.unique(x)
                critical_value = 0
                
                for j in classes_unique:
                    y_sub = pd.Series([y[k] for k in range(len(y)) if x[k]==j]) #creates a sub list of y for all rows in x that have class j
                    critical_value += (y_sub.size)*np.var(y_sub)
                    
                if(criteria_value==None or criteria_value>critical_value):
                    criteria_value = critical_value
                    attr_no = feature
                    split_value = None
            
            # If not categorical than our input features might be real.
            else:
                x_sorted = x.sort_values() #We sort the values of x so that we can split them accordingly.
                
                for j in range(len(y)-1):
                    index = x_sorted.index[j]
                    next_index = x_sorted.index[j+1]
                    splitvalue = (x[index]+x[next_index])/2 #find mean based on index and index+1
                    y_left = pd.Series([y[k] for k in range(y.size) if x[k]<=splitvalue])
                    y_right = pd.Series([y[k] for k in range(y.size) if x[k]>splitvalue])
                    critical_value = y_left.size*np.var(y_left) + y_right.size*np.var(y_right)
                    
                    if((criteria_value==None) or (critical_value<criteria_value)):
                        attr_no = feature
                        criteria_value = critical_value
                        split_value = splitvalue
    
    # If the present feature/current node is categorical
        if(split_value==None):
            current_node.attr_number_idx = attr_no
            current_node.isCategorical = True
            classes = np.unique(X[attr_no])
            
            for j in classes:
                y_update = pd.Series([y[k] for k in range(y.size) if X[attr_no][k]==j], dtype=y.dtype)
                x_update = X[X[attr_no]==j].reset_index().drop(['index',attr_no],axis=1)
                current_node.tree_child[j] = self.split_fit(x_update, y_update, depth+1)
                
        # If the current node is real type
        else:
            current_node.attr_number_idx = attr_no
            current_node.splitvalue = split_value
            y_update1 = pd.Series([y[k] for k in range(y.size) if X[attr_no][k]<=split_value], dtype=y.dtype)
            x_update1 = X[X[attr_no]<=split_value].reset_index().drop(['index'],axis=1)
            y_update2 = pd.Series([y[k] for k in range(y.size) if X[attr_no][k]>split_value], dtype=y.dtype)
            x_update2 = X[X[attr_no]>split_value].reset_index().drop(['index'],axis=1)
            current_node.tree_child["lessThan"] = self.split_fit(x_update1, y_update1, depth+1)
            current_node.tree_child["greaterThan"] = self.split_fit(x_update2, y_update2, depth+1)
        return current_node


    def fit(self, X, y):
        if(X.shape[0]==len(y) & len(y)>0): 
            self.head = self.split_fit(X,y,0)
        return self.head

    def predict(self, X):
        y_hat = []                  
        for i in range(X.shape[0]):
            xrow = X.iloc[i,:]    
            node = self.head
            while(not node.isleaf):                            
                if(node.isCategorical):                               
                    node = node.tree_child[xrow[node.attr_number_idx]]
                elif(xrow[node.attr_number_idx]>node.splitvalue):
                    node = node.tree_child["greaterThan"]
                else:
                    node = node.tree_child["lessThan"] 
            y_hat.append(node.val)                            
        y_hat = pd.Series(y_hat)
        return y_hat
    
    def plotTree(self, root, depth):
            if(root.isleaf):
                if(root.isCategorical):
                    return "Class "+str(root.val)
                else:
                    return "Value "+str(root.val)

            a = ""
            if(root.isCategorical):
                for i in root.tree_child.keys():
                    a += "?(X"+str(root.attr_number_idx)+" == "+str(i)+")\n" 
                    a += "\t"*(depth+1)
                    a += str(self.plotTree(root.tree_child[i], depth+1)).rstrip("\n") + "\n"
                    a += "\t"*(depth)
                a = a.rstrip("\t")
            else:
                a += "?(X"+str(root.attr_number_idx)+" <= "+str(root.splitvalue)+")\n"
                a += "\t"*(depth+1)
                a += "Y: " + str(self.plotTree(root.tree_child["lessThan"], depth+1)).rstrip("\n") + "\n"
                a += "\t"*(depth+1)
                a += "N: " + str(self.plotTree(root.tree_child["greaterThan"], depth+1)).rstrip("\n") + "\n"
        
            return a

# Class for Classification

class DecisionTreeClassifier(): 
    
    def __init__(self, criteria, max_depth=None):
        self.criteria = criteria 
        self.max_depth = max_depth
        self.head = None
        
    def split_fit(self,X,y,depth):
        current_node = Node()   # Creating a new Node
        current_node.attr_number_idx = -1
        split_value = None # for storing the threshold value/split
        criteria_value = None # for storing the final criteria value           
        classes=np.unique(y)

        #if no features exist
        if(X.shape[1]==0): 
                current_node.isleaf = True
                current_node.isCategorical = True
                current_node.val = y.value_counts().idxmax() # We return the mod of all the values present in/under that class
                return current_node

        # if only one feature is present
        if(len(classes)==1): 
                current_node.isleaf = True 
                current_node.isCategorical = True 
                current_node.val = classes[0]               # We return the the single class present
                return current_node

        # If we have exhausted the maximum depth
        if(self.max_depth!=None): 
                if(self.max_depth==depth): 
                    current_node.isleaf = True 
                    current_node.isCategorical = True
                    current_node.val = y.value_counts().idxmax() # We return the mod of all classes present.
                    return current_node
           
        for feature in X: # iterating through the columns in the dataframe
                
                x = X[feature] # selecting the column
                
                # If the given input is discrete
                if(x.dtype.name=="category"): 
                    critical_value = None
                    #Calculating the gini index
                    if(self.criteria=="gini_index"):        
                        classes1 = np.unique(x)
                        summ = 0
                        for j in classes1:
                            y_sub = pd.Series([y[k] for k in range(len(y)) if x[k]==j])
                            summ += (y_sub.size)*gini_index(y_sub)
                        critical_value = -1*(summ/x.size) 
                    #Calculating the Information Gain
                    else:                      
                        critical_value = information_gain(y,x)
                        
                    if((criteria_value==None) or (criteria_value<critical_value) ):
                            attr_no = feature
                            criteria_value = critical_value
                            split_value = None

                
                # For Real Input 
                else:
                    x_sorted = x.sort_values() #Sort based on values in column
                    for j in range(len(x_sorted)-1):
                        index = x_sorted.index[j]
                        next_index = x_sorted.index[j+1]

                        if(y[index]!=y[next_index]):
                            critical_value = None
                            splitvalue = (x[index]+x[next_index])/2 #for every index and index+1 , find the mean
                            
                            if(self.criteria=="information_gain"):         
                                info_attr = pd.Series(x<=splitvalue)
                                critical_value = information_gain(y,info_attr)
                                
                            else:                                              
                                y_left = pd.Series([y[k] for k in range(len(y)) if x[k]<=splitvalue])
                                y_right = pd.Series([y[k] for k in range(len(y)) if x[k]>splitvalue])
                                critical_value = y_left.size*gini_index(y_left) + y_right.size*gini_index(y_right)
                                critical_value =  -1*(critical_value/y.size)
                                
                            if((criteria_value==None) or (criteria_value<critical_value)):
                                attr_no = feature
                                criteria_value = critical_value
                                split_value = splitvalue
                                    
                if(split_value==None):
                
                    current_node.attr_number_idx = attr_no
                    current_node.isCategorical = True
                    classes = np.unique(X[attr_no])
                    
                    for j in classes:
                        y_update = pd.Series([y[k] for k in range(y.size) if X[attr_no][k]==j], dtype=y.dtype)
                        x_update = X[X[attr_no]==j].reset_index().drop(['index',attr_no],axis=1)
                        current_node.tree_child[j] = self.split_fit(x_update, y_update, depth+1)
                
                # current_node==split based
                else:
                    current_node.attr_number_idx = attr_no
                    current_node.splitvalue = split_value
                                   
                    y_update1 = pd.Series([y[k] for k in range(y.size) if X[attr_no][k]<=split_value], dtype=y.dtype)
                    x_update1 = X[X[attr_no]<=split_value].reset_index().drop(['index'],axis=1)
                    y_update2 = pd.Series([y[k] for k in range(y.size) if X[attr_no][k]>split_value], dtype=y.dtype)
                    x_update2 = X[X[attr_no]>split_value].reset_index().drop(['index'],axis=1)
                    current_node.tree_child["lessThan"] = self.split_fit(x_update1, y_update1, depth+1)
                    current_node.tree_child["greaterThan"] = self.split_fit(x_update2, y_update2, depth+1)
        return current_node    
        

    def fit(self, X, y):
        assert(y.size>0)
        assert(X.shape[0]==y.size)
        self.head = self.split_fit(X,y,0)
        return self.head
        
    def predict(self, X):
        y_hat = []                  

        for i in range(X.shape[0]):
            xrow = X.iloc[i,:]    #For every row in X
            node = self.head
            while(not node.isleaf):      #Check if node is not leaf                     
                if(node.isCategorical):       #Check if feature is categorical 
                    node = node.tree_child[xrow[node.attr_number_idx]]
                elif(xrow[node.attr_number_idx]>node.splitvalue):
                    node = node.tree_child["greaterThan"]
                else:
                    node = node.tree_child["lessThan"]

                # else:                         # Feature is real            
                #     if(xrow[node.attr_number_idx]>node.splitvalue):
                #         node = node.tree_child["greaterThan"]
                #     else:
                #         node = node.tree_child["lessThan"]
            
            y_hat.append(node.val)                           
        
        y_hat = pd.Series(y_hat)
        #print(pd.Series(y_hat))
        return y_hat
    
    def plotTree(self, root, depth):
            if(root.isleaf):
                if(root.isCategorical):
                    return "Class "+str(root.val)
                else:
                    return "Value "+str(root.val)

            a = ""
            if(root.isCategorical):
                for i in root.tree_child.keys():
                    a += "?(X"+str(root.attr_number_idx)+" == "+str(i)+")\n" 
                    a += "\t"*(depth+1)
                    a += str(self.plotTree(root.tree_child[i], depth+1)).rstrip("\n") + "\n"
                    a += "\t"*(depth)
                a = a.rstrip("\t")
            else:
                a += "?(X"+str(root.attr_number_idx)+" <= "+str(root.splitvalue)+")\n"
                a += "\t"*(depth+1)
                a += "Y: " + str(self.plotTree(root.tree_child["lessThan"], depth+1)).rstrip("\n") + "\n"
                a += "\t"*(depth+1)
                a += "N: " + str(self.plotTree(root.tree_child["greaterThan"], depth+1)).rstrip("\n") + "\n"
        
            return a
    
class DecisionTree():
    def __init__(self, criterion="information_gain", max_depth=None):
        self.criterion = criterion
        self.max_depth = max_depth
        self.head = None
        self.tree=None
    


    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Function to train and construct the decision tree
        """
        if(y.dtype.name=="category"):
            self.tree =DecisionTreeClassifier(criteria=self.criterion, max_depth=self.max_depth)
            self.head = self.tree.fit(X, y)
        else:
            self.tree =DecisionTreeRegressor(criteria=self.criterion, max_depth=self.max_depth) #Split based on Inf. Gain
            self.head = self.tree.fit(X, y)

        

    def predict(self, X: pd.DataFrame, max_depth=np.inf) -> pd.Series:
        """
        Funtion to run the decision tree on test inputs
        """

        y_hat = self.tree.predict(X)                 
        return y_hat

    def plot(self) -> None:
        """
        Function to plot the tree

        Output Example:
        ?(X1 > 4)
            Y: ?(X2 > 7)
                Y: Class A
                N: Class B
            N: Class C
        Where Y => Yes and N => No
        """
        plot1= self.tree.plotTree(self.head,depth=0)
        print(plot1)
