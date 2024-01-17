
import enum
from random import sample
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree as skl_tree

class AdaBoostClassifier():
    def __init__(self, base_estimator=DecisionTreeClassifier, n_estimators=3, max_depth=1, criterion="entropy"): # Optional Arguments: Type of estimator
        '''
        :param base_estimator: The base estimator model instance from which the boosted ensemble is built (e.g., DecisionTree, LinearRegression).
                               If None, then the base estimator is DecisionTreeClassifier(max_depth=1).
                               You can pass the object of the estimato class
        :param n_estimators: The maximum number of estimators at which boosting is terminated. In case of perfect fit, the learning procedure may be stopped early.
        '''
        self.base_estimator = base_estimator
        self.max_depth= max_depth
        self.criterion = criterion
        self.n_estimators = n_estimators
        self.models= []
        self.alphas = []

    def fit(self, X, y):
        """
        Function to train and construct the AdaBoostClassifier
        Inputs:
        X: pd.DataFrame with rows as samples and columns as features (shape of X is N X P) where N is the number of samples and P is the number of columns.
        y: pd.Series with rows corresponding to output variable (shape of Y is N)
        """
        n= len(y)
        weights = np.ones(n)/n

        for t in range(self.n_estimators):
            clf = DecisionTreeClassifier(max_depth=self.max_depth)
            clf.fit(X,y, sample_weight=weights)
            clf_pred = pd.Series(clf.predict(X))

            mis_class = clf_pred!=y
            err = np.sum(weights[mis_class])/np.sum(weights)
            alpha_1 = np.log((1-err)/err)*0.5

            weights[mis_class] *= np.exp(alpha_1)
            weights[~mis_class]*= np.exp(-alpha_1)

            self.models.append(clf)
            self.alphas.append(alpha_1)



    def predict(self, X):
        """
        Input:
        X: pd.DataFrame with rows as samples and columns as features
        Output:
        y: pd.Series with rows corresponding to output variable. THe output variable in a row is the prediction for sample in corresponding row in X.
        """
        final_pred= np.zeros(X.shape[0])
        d = {}
        y_predicted_list = []
        alpha_list = []
        for i, (alpha_1, clf) in enumerate(zip(self.alphas,self.models)):
            y_predicted = list(clf.predict(X))
            # alphas = [alpha_1]*len(y_predicted)
            y_predicted_list.append(y_predicted)
            alpha_list.append(alpha_1)
        y_predicted_df = pd.DataFrame(y_predicted_list)
        final_pred = []
        for i in y_predicted_df.columns:
            distinct_pred = y_predicted_df[i].unique()
            max = 0
            id = 0
            for j in distinct_pred:
                summ = 0
                for k in y_predicted_df.index:
                    if(y_predicted_df[i][k] ==j ):
                        summ+=alpha_list[k]
                if(summ>max):
                    max = summ
                    id = j
            final_pred.append(id) 
        return pd.Series(final_pred)

    def plot(self,X,y):
        """
        Function to plot the decision surface for AdaBoostClassifier for each estimator(iteration).
        Creates two figures
        Figure 1 consists of 1 row and `n_estimators` columns
        The title of each of the estimator should be associated alpha (similar to slide#38 of course lecture on ensemble learning)
        Further, the scatter plot should have the marker size corresponnding to the weight of each point.

        Figure 2 should also create a decision surface by combining the individual estimators

        Reference for decision surface: https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html

        This function should return [fig1, fig2]
        """
        # assert(len(list(X.columns))==2)
        color = ["r","g","b"]
        Zs= None

        fig1, ax = plt.subplots(1, len(self.models), figsize= (5*len(self.models),4))
        feat_min, feat_max= X.iloc[:,0].min(), X.iloc[:,0].max()
        feat2_min, feat2_max = X.iloc[:,1].min(), X.iloc[:,1].max()
        x_range = feat_max-feat_min
        y_range = feat2_max-feat2_min

        for i, (alpha_1,clf) in enumerate(zip(self.alphas,self.models)):
            print("Model Number: {}".format(i+1))
            print(skl_tree.export_text(clf))
            xx, yy = np.meshgrid(np.arange(feat_min-0.2, feat_max+0.2, (x_range)/50),
                                 np.arange(feat2_min-0.2, feat2_max+0.2, (y_range)/50))

            ax[i].set_xlabel("X1")       
            ax[i].set_ylabel("X2")

            Z= clf.predict(np.c_[xx.ravel(),yy.ravel()])
            Z= Z.reshape(xx.shape)
            
            if Zs is None:
                Zs= alpha_1*Z
            else:
                Zs+= alpha_1*Z

            cs= ax[i].contourf(xx,yy,Z,cmap=plt.cm.RdYlBu) 
            fig1.colorbar(cs, ax=ax[i],shrink =0.9)
            for label in y.unique():
                idx= (y==label)
                # id = [(y.cat.categories).index(y[idx].iloc[0])]
                ax[i].scatter(X[idx].iloc[:,0],X[idx].iloc[:, 1],cmap=plt.cm.RdYlBu, edgecolor='black', s=30,
                               label="Class: "+str(label))   
            ax[i].set_title("Decision Surface Tree: " + str(i+1))
            ax[i].legend() 
        fig1.tight_layout()               

        #plot 2
        fig2, ax2 = plt.subplots(1, 1, figsize=(5, 4))
        com_surface = np.sign(Zs)
        cs = ax2.contourf(xx, yy, com_surface, cmap=plt.cm.RdYlBu)
        for label in y.unique():
            idx = (y == label)
            # id = list(y.cat.categories).index(y[idx].iloc[0])
            ax2.scatter(X[idx].iloc[:, 0], X[idx].iloc[:, 1],
                        cmap=plt.cm.RdYlBu, edgecolor='black', s=30,
                        label="Class: "+str(label))
        ax2.set_ylabel("X2")
        ax2.set_xlabel("X1")
        ax2.legend(loc="lower right")
        ax2.set_title("Common Decision Surface")
        fig2.colorbar(cs, ax=ax2, shrink=0.9)

        plt.show()

        # Saving Figures
        fig1.savefig(os.path.join("figures", "Q3_Fig1.png"))
        fig2.savefig(os.path.join("figures", "Q3_Fig2.png"))

        return fig1, fig2


