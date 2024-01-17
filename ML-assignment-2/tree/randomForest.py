import enum
from .base import DecisionTree
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
import pandas as pd
import numpy as np 
import random
import matplotlib.pyplot as plt
import os
from sklearn.tree import  plot_tree
np.random.seed(42)

def split_reg_data(X_r,y_r):
    total_vars = (len(list(X_r.columns)))

    drop_vars = []
    for i in range(int(total_vars/2)):
        drop_vars.append(random.randint(0,total_vars-1))

    X_r = X_r.drop(drop_vars, axis=1)

    X_r['y'] = y_r

    sample_lst = []
    for i in range(len(X_r)):
        X_sampled = X_r.sample(n=1)
        sample_lst.append(X_sampled)

    X_r = pd.concat(sample_lst)
    X_r.reset_index(drop=True,inplace = True)

    y_r = X_r.pop('y')

    return X_r,y_r,X_r.columns

class RandomForestClassifier():
    def __init__(self, n_estimators=100, criterion='gini', max_depth=None):
        '''
        :param estimators: DecisionTree
        :param n_estimators: The number of trees in the forest.
        :param criterion: The function to measure the quality of a split.
        :param max_depth: The maximum depth of the tree.
        '''
        self.max_depth = max_depth
        self.n_estimators = n_estimators
        if criterion=="gini":
            self.criterion = criterion
        else:
            self.criterion = "entropy"
        self.clf=[]
        self.X_split=[]
        self.y_split=[]

        self.input1=[]
        self.op_labels=[]

        pass
    def split_data(self,X_s,y_s):

        X_s['y'] = y_s

        sample_lst = []
        for i in range(len(X_s)):
            X_sampled = X_s.sample(n=1)
            sample_lst.append(X_sampled)
    
        X_s = pd.concat(sample_lst)
        X_s.reset_index(drop=True,inplace = True)

        y_s = X_s.pop('y')

        return X_s,y_s

    def fit(self, X, y):
        """
        Function to train and construct the RandomForestClassifier
        Inputs:
        X: pd.DataFrame with rows as samples and columns as features (shape of X is N X P) where N is the number of samples and P is the number of columns.
        y: pd.Series with rows corresponding to output variable (shape of Y is N)
        """
        self.input1= X
        self.op_labels = y
        for n in range(self.n_estimators):
            X_new, y_new = self.split_data(X,y)
            clasifier = DecisionTreeClassifier(criterion=self.criterion,max_depth=4)
            clasifier.fit(X_new,y_new)

            self.clf.append(clasifier)
            self.X_split.append(X_new)
            self.y_split.append(y_new)

        pass

    def predict(self, X):
        """
        Funtion to run the RandomForestClassifier on a data point
        Input:
        X: pd.DataFrame with rows as samples and columns as features
        Output:
        y: pd.Series with rows corresponding to output variable. THe output variable in a row is the prediction for sample in corresponding row in X.
        """
        out = "y"
        if (isinstance(X, pd.DataFrame)):
            if (out in X.columns):
                X = X.drop(["y"],axis=1)

        y_pred = []
        predictions = []
        for claf in self.clf:
            temp = claf.predict(X)
            predictions.append(temp)

        pred_arr = (np.array(predictions)).T
    
        y_pred = [np.unique(i)[np.argmax(np.unique(i, return_counts=True)[1])] for i in pred_arr]
        return(pd.Series(y_pred))

    def plot(self,X,y):
        """
        Function to plot for the RandomForestClassifier.
        It creates three figures

        1. Creates a figure with 1 row and `n_estimators` columns. Each column plots the learnt tree. If using your sklearn, this could a matplotlib figure.
        If using your own implementation, it could simply invole print functionality of the DecisionTree you implemented in assignment 1 and need not be a figure.

        2. Creates a figure showing the decision surface for each estimator

        3. Creates a figure showing the combined decision surface

        """

        for i in range(self.n_estimators):
            plot_tree(self.clf[i])
            plt.title(f'Tree Number{i}')
            plt.show()
        
        
        plot_colors = ["red","green","yellow","#D95319","#EDB120","#77AC30"]
        plot_step = 0.02
        n_classes = 5
        for _ in range (self.n_estimators):
            plt.subplot(2, 5, _+1 )
            x_min, x_max = self.X_split[_].iloc[:, 0].min() - 1, self.X_split[_].iloc[:, 0].max() + 1
            y_min, y_max = self.X_split[_].iloc[:, 1].min() - 1, self.X_split[_].iloc[:, 1].max() + 1
            xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),np.arange(y_min, y_max, plot_step))
            # print(xx,yy)
            plt.tight_layout(h_pad=0.5, w_pad=0.5, pad=2.5)
            Z = self.clf[_].predict(np.c_[xx.ravel(), yy.ravel(),[self.X_split[_].iloc[:,2].mean()]*len(xx.ravel()), [self.X_split[_].iloc[:,3].mean()]*len(xx.ravel()), [self.X_split[_].iloc[:,4].mean()]*len(xx.ravel())])
            Z = Z.reshape(xx.shape)
            cs = plt.contourf(xx, yy, Z, cmap=plt.cm.RdYlGn)
            for i, color in zip(range(n_classes), plot_colors):
                idx = np.where(self.y_split[_] == i)
                for i in range (len(idx[0])):
                    plt.scatter(self.X_split[_].loc[idx[0][i]][0], self.X_split[_].loc[idx[0][i]][1],c=color, edgecolor='black', s=15)
                    plt.ylabel("X2")
                    plt.xlabel("X1")
        plt.suptitle("Decision surface of a tree using two features")
        plt.legend(loc='lower right', borderpad=0, handletextpad=0)
        plt.axis("tight")
        
        plt.show()
        fig1 = plt

        # Figure 2
        print("Combined decision surface ")
        plot_colors = ["red","green","yellow","#D95319","#EDB120","#77AC30"]
        plot_step = 0.02
        n_classes = 5
        x_min, x_max = self.input1.iloc[:, 0].min() - 1, self.input1.iloc[:, 0].max() + 1
        y_min, y_max = self.input1.iloc[:, 1].min() - 1, self.input1.iloc[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),np.arange(y_min, y_max, plot_step))
        plt.tight_layout(h_pad=0.5, w_pad=0.5, pad=2.5)
        Z = self.predict(np.c_[xx.ravel(), yy.ravel(), [self.input1.iloc[:,2].mean()]*len(xx.ravel()), [self.input1.iloc[:,3].mean()]*len(xx.ravel()), [self.input1.iloc[:,4].mean()]*len(xx.ravel())])
        Z = np.array(Z)
        Z = Z.reshape(xx.shape)
        cs = plt.contourf(xx, yy, Z, cmap=plt.cm.RdYlGn)
        for i, color in zip(range(n_classes), plot_colors):
            idx = np.where(self.op_labels == i)
            for i in range (len(idx[0])):
                plt.scatter(self.input1.loc[idx[0][i]][0], self.input1.loc[idx[0][i]][1],c=color,cmap=plt.cm.PiYG, edgecolor='black', s=15)
        plt.suptitle("Decision surface by combining all the estimators")
        plt.ylabel("X2")
        plt.xlabel("X1")
        plt.legend(loc='lower right', borderpad=0, handletextpad=0)
        plt.axis("tight")
        plt.show()
        fig2 = plt

        return fig1, fig2



class RandomForestRegressor():
    def __init__(self, n_estimators=100, criterion='variance', max_depth=None):
        '''
        :param n_estimators: The number of trees in the forest.
        :param criterion: The function to measure the quality of a split.
        :param max_depth: The maximum depth of the tree.
        '''
        
        self.criterion="squared_error"
        self.n_estimators = n_estimators
        self.reg_list = []
        self.max_depth = max_depth
        self.X_split=[]
        self.y_split=[]
        self.input1=None
        self.op_labels = None
        self.feat_list=[]

    def fit(self, X, y):
        """
        Function to train and construct the RandomForestRegressor
        Inputs:
        X: pd.DataFrame with rows as samples and columns as features (shape of X is N X P) where N is the number of samples and P is the number of columns.
        y: pd.Series with rows corresponding to output variable (shape of Y is N)
        """
        out="y"
        self.input1=X
        self.op_labels=y
        if (isinstance(X, pd.DataFrame)):
            if (out in X.columns):
                X = X.drop(["y"],axis=1)
        for n in range(self.n_estimators):
            X_sampled,y_sampled,X_r_cols = split_reg_data(X,y)
            tree = DecisionTreeRegressor()
            tree.fit(X_sampled, y_sampled)
            self.reg_list.append(tree)
            self.X_split.append(X_sampled)
            self.y_split.append(y_sampled)
            self.feat_list.append(X_r_cols)
        for i in range(len(self.feat_list)):
            self.feat_list[i]= list(map(int, self.feat_list[i]))
        pass

    def predict(self, X):
        """
        Funtion to run the RandomForestRegressor on a data point
        Input:
        X: pd.DataFrame with rows as samples and columns as features
        Output:
        y: pd.Series with rows corresponding to output variable. THe output variable in a row is the prediction for sample in corresponding row in X.
        """
        y_pred = np.zeros(len(X))
        pred=[]
        for i,reg in enumerate(self.reg_list):
            pred.append(reg.predict(X[self.feat_list[i]]))
        pred_arr = (np.array(pred)).T
        y_pred = [np.mean(i) for i in pred_arr]
        return(np.array(y_pred))

    def plot(self):
        """
        Function to plot for the RandomForestClassifier.
        It creates three figures

        1. Creates a figure with 1 row and `n_estimators` columns. Each column plots the learnt tree. If using your sklearn, this could a matplotlib figure.
        If using your own implementation, it could simply invole print functionality of the DecisionTree you implemented in assignment 1 and need not be a figure.

        2. Creates a figure showing the decision surface/estimation for each estimator. Similar to slide 9, lecture 4

        3. Creates a figure showing the combined decision surface/prediction

        """
        for i in range(self.n_estimators):
            plot_tree(self.reg_list[i])
            plt.title(f'Tree Number{i}')
            plt.show()

        pass
















