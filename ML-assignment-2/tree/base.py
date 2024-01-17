"""
The current code given is for the Assignment 1.
You will be expected to use this to make trees for:
> discrete input, discrete output
> real input, real output
> real input, discrete output
> discrete input, real output
"""

import numpy as np
import os 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import tree
from .utils import entropy, information_gain, gini_index
from matplotlib.colors import ListedColormap

np.random.seed(42)

class Node():
    def __init__(self):
        self.val = None #to store parameter value
        self.isedgeleaf = False #to check if node is leaf
        self.attr_number = None #to store feature number of split
        self.splitvalue = None # to store split value
        self.isAttritbute= False # True if categorical data
        self.tree_child = {} # Dict to store children 

# Class for Regression

class DecisionTreeRegressor(): 
    
    def __init__(self, criteria, max_depth=None):
        self.critera = criteria
        self.max_depth = max_depth
        self.head = None
        
    def split_fit(self,X,y,depth,weights=None):
        curr_Node = Node()   # Creating a new Node
        curr_Node.attr_number = -1
        splitval = None # Variable to store value of split
        criteria_val = None # Variable to store final criteria value      
          
        classes=np.unique(y) 
        
        #if zero features exist
        if(X.shape[1]==0):
            curr_Node.isedgeleaf = True
            curr_Node.val = y.mean()
            return curr_Node

        #if max_depth is specified
        if(self.max_depth!=None):
            if(depth==self.max_depth):
                curr_Node.isedgeleaf = True
                curr_Node.val = y.mean()
                return curr_Node
        
        if(len(classes)==1): 
            curr_Node.isedgeleaf = True
            curr_Node.val = classes[0]
            return curr_Node

        for feature in X:
            x = X[feature]

            # For discrete input
            if(x.dtype.name=="category"):
                classes_unique = np.unique(x)
                crit_val = 0
                
                for j in classes_unique:
                    y_sub = pd.Series([y[k] for k in range(len(y)) if x[k]==j]) #creates a sub list of y for all rows in x that have class j
                    crit_val += (y_sub.size)*np.var(y_sub)
                    
                if(criteria_val==None):
                    criteria_val = crit_val
                    attr_no = feature
                    splitval = None
                else:
                    if(criteria_val>crit_val):
                        criteria_val = crit_val
                        attr_no = feature
                        splitval = None
            
            # Real Input 
            else:
                x_sorted = x.sort_values() #sort values of x
                
                for j in range(len(y)-1):
                    index = x_sorted.index[j]
                    next_index = x_sorted.index[j+1]
                    splitvalue = (x[index]+x[next_index])/2 #find mean based on index and index+1
                    y_sub1 = pd.Series([y[k] for k in range(y.size) if x[k]<=splitvalue])
                    y_sub2 = pd.Series([y[k] for k in range(y.size) if x[k]>splitvalue])
                    crit_val = y_sub1.size*np.var(y_sub1) + y_sub2.size*np.var(y_sub2)
                    
                    if(criteria_val==None):
                        attr_no = feature
                        criteria_val = crit_val
                        splitval = splitvalue
                    else:
                        if(crit_val<criteria_val):
                            attr_no = feature
                            criteria_val = crit_val
                            splitval = splitvalue
    

    # If current attribute is categorical 
        if(splitval==None):
            
            curr_Node.attr_number = attr_no
            curr_Node.isAttritbute = True
            classes = np.unique(X[attr_no])
            
            for j in classes:
                y_new = pd.Series([y[k] for k in range(y.size) if X[attr_no][k]==j], dtype=y.dtype)
                X_new = X[X[attr_no]==j].reset_index().drop(['index',attr_no],axis=1)
                weights_new = weights[X[attr_no]==j].reset_index().drop(['index'],axis=1)
                curr_Node.tree_child[j] = self.split_fit(X_new, y_new, depth+1,weights_new)
                
        # if current node feature is real 
        else:
            curr_Node.attr_number = attr_no
            curr_Node.splitvalue = splitval
            y_new1 = pd.Series([y[k] for k in range(y.size) if X[attr_no][k]<=splitval], dtype=y.dtype)
            X_new1 = X[X[attr_no]<=splitval].reset_index().drop(['index'],axis=1)
            weights_new1 = weights[X[attr_no]<=splitval].reset_index().drop(['index'],axis=1)
            y_new2 = pd.Series([y[k] for k in range(y.size) if X[attr_no][k]>splitval], dtype=y.dtype)
            X_new2 = X[X[attr_no]>splitval].reset_index().drop(['index'],axis=1)
            weights_new2 = weights[X[attr_no]>splitval].reset_index().drop(['index'],axis=1)
            curr_Node.tree_child["lessThan"] = self.split_fit(X_new1, y_new1, depth+1,weights_new1)
            curr_Node.tree_child["greaterThan"] = self.split_fit(X_new2, y_new2, depth+1,weights_new2)
        return curr_Node


    def fit(self, X, y, weights=None):
        if(weights is None):
            weights = pd.Series(np.ones(X.shape[0]))
        if(X.shape[0]==len(y) & len(y)>0): 
            self.head = self.split_fit(X,y,0,weights)
        return self.head

    def predict(self, X):
        y_hat = []                  

        for i in range(X.shape[0]):
            xrow = X.iloc[i,:]    
            node = self.head
            while(not node.isedgeleaf):                            
                if(node.isAttritbute):                               
                    node = node.tree_child[xrow[node.attr_number]]
                else:                                       
                    if(xrow[node.attr_number]>node.splitvalue):
                        node = node.tree_child["greaterThan"]
                    else:
                        node = node.tree_child["lessThan"]
            
            y_hat.append(node.val)                           
        
        y_hat = pd.Series(y_hat)
        # print(pd.Series(y_hat))
        return y_hat
    
    def plotTree(self, root, depth):
            if(root.isedgeleaf):
                if(root.isAttritbute):
                    return "Class "+str(root.val)
                else:
                    return "Value "+str(root.val)

            a = ""
            if(root.isAttritbute):
                for i in root.tree_child.keys():
                    a += "?(X"+str(root.attr_number)+" == "+str(i)+")\n" 
                    a += "\t"*(depth+1)
                    a += str(self.plotTree(root.tree_child[i], depth+1)).rstrip("\n") + "\n"
                    a += "\t"*(depth)
                a = a.rstrip("\t")
            else:
                a += "?(X"+str(root.attr_number)+" <= "+str(root.splitvalue)+")\n"
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
        
    def split_fit(self,X,y,depth,weights):
        curr_Node = Node()   # Creating a new Node
        curr_Node.attr_number = -1
        splitval = None #To store split value
        criteria_val = None # To store final criteria value           
        classes=np.unique(y)

        #if zero features exist
        if(X.shape[1]==0): 
                curr_Node.isedgeleaf = True
                curr_Node.isAttritbute = True
                curr_Node.val = y.value_counts().idxmax() # Mode of all outputs 
                return curr_Node

        # if a single value of y exists
        if(len(classes)==1): 
                curr_Node.isedgeleaf = True 
                curr_Node.isAttritbute = True 
                curr_Node.val = classes[0] 
                return curr_Node

        #if max depth is specified
        if(self.max_depth!=None): 
                if(self.max_depth==depth): 
                    curr_Node.isedgeleaf = True 
                    curr_Node.isAttritbute = True
                    curr_Node.val = y.value_counts().idxmax() #Output is the max number of output in y
                    return curr_Node
           
        for feature in X: # i is column name in X dataframe
                
                x = X[feature] #x_column is column X[i]
                
                #for discrete input
                if(x.dtype.name=="category"): 
                    crit_val = None
                    #Calculating the gini index
                    if(self.criteria=="gini_index"):        
                        classes1 = np.unique(x)
                        s1 = 0
                        for j in classes1:
                            y_sub = pd.Series([y[k] for k in range(len(y)) if x[k]==j])
                            s1 += (y_sub.size)*gini_index(y_sub)
                        crit_val = -1*(s1/x.size) 
                    #Calculating the Information Gain
                    else:                      
                        crit_val = information_gain(y,x,weights)
                        
                    if(criteria_val==None):
                            attr_no = feature
                            criteria_val = crit_val
                            splitval = None
                    else:
                        #Choosing the feature with max info gain
                        if(criteria_val<crit_val):
                            attr_no = feature
                            criteria_val = crit_val
                            splitval = None
                
                # For Real Input 
                else:
                    x_sorted = x.sort_values() #Sort based on values in column
                    for j in range(len(x_sorted)-1):
                        index = x_sorted.index[j]
                        next_index = x_sorted.index[j+1]

                        if(y[index]!=y[next_index]):
                            crit_val = None
                            splitvalue = (x[index]+x[next_index])/2 #for every index and index+1 , find the mean
                            
                            if(self.criteria=="information_gain"):         
                                info_attr = pd.Series(x<=splitvalue)
                                crit_val = information_gain(y,info_attr,weights)
                                
                            else:                                              
                                y_sub1 = pd.Series([y[k] for k in range(len(y)) if x[k]<=splitvalue])
                                y_sub2 = pd.Series([y[k] for k in range(len(y)) if x[k]>splitvalue])
                                crit_val = y_sub1.size*gini_index(y_sub1) + y_sub2.size*gini_index(y_sub2)
                                crit_val =  -1*(crit_val/y.size)
                                
                            if(criteria_val==None):
                                attr_no = feature
                                criteria_val = crit_val
                                splitval = splitvalue
                            else:
                                if(criteria_val<crit_val):
                                    attr_no = feature
                                    criteria_val = crit_val
                                    splitval = splitvalue
                                    
                if(splitval==None):
                
                    curr_Node.attr_number = attr_no
                    curr_Node.isAttritbute = True
                    classes = np.unique(X[attr_no])
                    
                    for j in classes:
                        y_new = pd.Series([y[k] for k in range(y.size) if X[attr_no][k]==j], dtype=y.dtype)
                        X_new = X[X[attr_no]==j].reset_index().drop(['index',attr_no],axis=1)
                        curr_Node.tree_child[j] = self.split_fit(X_new, y_new, depth+1,weights)
                
                # curr_Node==split based
                else:
                    curr_Node.attr_number = attr_no
                    curr_Node.splitvalue = splitval
                                   
                    y_new1 = pd.Series([y[k] for k in range(y.size) if X[attr_no][k]<=splitval], dtype=y.dtype)
                    X_new1 = X[X[attr_no]<=splitval].reset_index().drop(['index'],axis=1)
                    weights_new1 = weights[X[attr_no]<=splitval]
                    y_new2 = pd.Series([y[k] for k in range(y.size) if X[attr_no][k]>splitval], dtype=y.dtype)
                    X_new2 = X[X[attr_no]>splitval].reset_index().drop(['index'],axis=1)
                    weights_new2 = weights[X[attr_no]>splitval]
                    curr_Node.tree_child["lessThan"] = self.split_fit(X_new1, y_new1, depth+1,weights_new1)
                    curr_Node.tree_child["greaterThan"] = self.split_fit(X_new2, y_new2, depth+1,weights_new2)
        return curr_Node    
        

    def fit(self, X, y,weights=None):
        assert(y.size>0)
        assert(X.shape[0]==y.size)
        if(weights is None):
            weights = pd.Series(np.ones(X.shape[0]))
        self.head = self.split_fit(X,y,0,weights)
        return self.head
        
    def predict(self, X):
        y_hat = []                  

        for i in range(X.shape[0]):
            xrow = X.iloc[i,:]    #For every row in X
            node = self.head
            while(not node.isedgeleaf):      #Check if node is not leaf                     
                if(node.isAttritbute):       #Check if feature is categorical 
                    node = node.tree_child[xrow[node.attr_number]]
                else:                         # Feature is real            
                    if(xrow[node.attr_number]>node.splitvalue):
                        node = node.tree_child["greaterThan"]
                    else:
                        node = node.tree_child["lessThan"]
            
            y_hat.append(node.val)                           
        
        y_hat = pd.Series(y_hat)
        #print(pd.Series(y_hat))
        return y_hat
    
    def plotTree(self, root, depth):
            if(root.isedgeleaf):
                if(root.isAttritbute):
                    return "Class "+str(root.val)
                else:
                    return "Value "+str(root.val)

            a = ""
            if(root.isAttritbute):
                for i in root.tree_child.keys():
                    a += "?(X"+str(root.attr_number)+" == "+str(i)+")\n" 
                    a += "\t"*(depth+1)
                    a += str(self.plotTree(root.tree_child[i], depth+1)).rstrip("\n") + "\n"
                    a += "\t"*(depth)
                a = a.rstrip("\t")
            else:
                a += "?(X"+str(root.attr_number)+" <= "+str(root.splitvalue)+")\n"
                a += "\t"*(depth+1)
                a += "Y: " + str(self.plotTree(root.tree_child["lessThan"], depth+1)).rstrip("\n") + "\n"
                a += "\t"*(depth+1)
                a += "N: " + str(self.plotTree(root.tree_child["greaterThan"], depth+1)).rstrip("\n") + "\n"
        
            return a
        


class DecisionTree():
    def __init__(self, criterion="information_gain", max_depth=None):
        """
        Put all infromation to initialize your tree here.
        Inputs:
        > criterion : {"information_gain", "gini_index"} # criterion won't be used for regression
        > max_depth : The maximum depth the tree can grow to 
        """
        self.criterion = criterion
        self.max_depth = max_depth
        self.head = None
        self.tree=None

    def fit(self, X, y, weights=None):
        """
        Function to train and construct the decision tree
        Inputs:
        X: pd.DataFrame with rows as samples and columns as features (shape of X is N X P) where N is the number of samples and P is the number of columns.
        y: pd.Series with rows corresponding to output variable (shape of Y is N)
        """
        if(y.dtype.name=="category"):
            self.tree =DecisionTreeClassifier(criteria=self.criterion, max_depth=self.max_depth)
            self.head = self.tree.fit(X, y)
        else:
            self.tree =DecisionTreeRegressor(criteria=self.criterion, max_depth=self.max_depth) #Split based on Inf. Gain
            self.head = self.tree.fit(X, y)


    def predict(self, X):
        """
        Funtion to run the decision tree on a data point
        Input:
        X: pd.DataFrame with rows as samples and columns as features
        Output:
        y: pd.Series with rows corresponding to output variable. THe output variable in a row is the prediction for sample in corresponding row in X.
        """
        y_hat = self.tree.predict(X)                 
        return y_hat
        

    def plot(self,X,y,clf):
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
        # print(plot1)
        cm= plt.cm.RdBu
        h=0.02
        cm_bright = ListedColormap(["darkorange","lightseagreen"])
        ax1= plt.subplot(1,1,1)
        x= pd.DataFrame(X)
        y= pd.Series(y)
        x_min,x_max = x[x.columns[0]].min()-0.5 , x[x.columns[0]].max()+0.5 
        y_min,y_max = x[x.columns[1]].min()-0.5 , x[x.columns[1]].max()+0.5
        xx,yy = np.meshgrid(np.arange(x_min,x_max,h),np.arange(y_min,y_max,h))
        Z = np.array(clf.predict(pd.DataFrame(np.c_[xx.ravel(), yy.ravel()],columns= x.columns)))
        Z = Z.reshape(xx.shape)
        ax1.contourf(xx, yy, Z, cmap=cm, alpha=.8)
        ax1.scatter(x[x.columns[0]], x[x.columns[1]], c = y, cmap=cm_bright,edgecolors='k', alpha=0.6)
        ax1.set_xlim(xx.min(), xx.max())
        ax1.set_ylim(yy.min(), yy.max())
        plt.savefig(os.path.join("figures", "Q2_Fig1.png"))
        plt.show()
        # plt.title("Estimator "+ str(i+1))