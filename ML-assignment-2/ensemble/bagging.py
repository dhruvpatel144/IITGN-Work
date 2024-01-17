from tree.base import DecisionTree
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn import tree as sktree
import matplotlib.pyplot as plt
from sklearn.utils.extmath import weighted_mode
from joblib import Parallel, delayed
import os


class BaggingClassifier():
    def __init__(self, base_estimator=DecisionTree, n_estimators=5,
                 max_depth=100, criterion="information_gain", n_jobs = 1):
        '''
        :param base_estimator: The base estimator model instance from which the bagged ensemble is built (e.g., DecisionTree(), LinearRegression()).
                               You can pass the object of the estimator class
        :param n_estimators: The number of estimators/models in ensemble.
        '''
        self.base_estimator = base_estimator
        self.max_depth = max_depth
        self.criterion = criterion
        self.n_estimators = n_estimators
        self.trees = []
        self.datas = []
        self.n_jobs = n_jobs


    def fit(self, X, y):
        """
        Function to train and construct the BaggingClassifier
        Inputs:
        X: pd.DataFrame with rows as samples and columns as features (shape of X is N X P) where N is the number of samples and P is the number of columns.
        y: pd.Series with rows corresponding to output variable (shape of Y is N)
        """
        def fit_tree(X, y):
            A = []
            X_sub_data = X.sample(frac=1, axis='rows', replace=True)
            y_sub_data = y[X_sub_data.index]
            X_sub_data = X_sub_data.reset_index(drop=True)
            y_sub_data = y_sub_data.reset_index(drop=True)

            # Learning new tree on sampled data
            tree = self.base_estimator(criterion=self.criterion)
            # print(X_sub_data, y_sub_data)
            tree.fit(X_sub_data, y_sub_data)

            # Storing data and tree
            A.append(tree)
            A.append([X_sub_data, y_sub_data])
            return A
        if self.n_jobs == 1:
            for n in tqdm(range(self.n_estimators)):
                # Sampling data
                X_sub_data = X.sample(frac=1, axis='rows', replace=True)
                y_sub_data = y[X_sub_data.index]
                X_sub_data = X_sub_data.reset_index(drop=True)
                y_sub_data = y_sub_data.reset_index(drop=True)

                # Learning new tree on sampled data
                tree = self.base_estimator(criterion=self.criterion)
                tree.fit(X_sub_data, y_sub_data)

                # Storing data and tree
                self.trees.append(tree)
                self.datas.append([X_sub_data, y_sub_data])
        else:
            result = Parallel(n_jobs=self.n_jobs, prefer = "threads")(
                delayed(fit_tree)(X, y)
                for i in range(self.n_estimators))
            # print(result)
            for j in result:
                self.trees.append(j[0]) 
                self.datas.append(j[1])
                print(j[0])
                print("break")
                print(j[1])
                print("break")

    def predict(self, X):
        """
        Funtion to run the BaggingClassifier on a data point
        Input:
        X: pd.DataFrame with rows as samples and columns as features
        Output:
        y: pd.Series with rows corresponding to output variable. THe output variable in a row is the prediction for sample in corresponding row in X.
        """
        y_hat_total = None
        for i, tree in enumerate(self.trees):
            if y_hat_total is None:
                y_hat_total = pd.Series(tree.predict(X)).to_frame()
            else:
                y_hat_total[i] = tree.predict(X)
        return y_hat_total.mode(axis=1)[0]

    def plot(self, X, y):
        """
        Function to plot the decision surface for BaggingClassifier for each estimator(iteration).
        Creates two figures
        Figure 1 consists of 1 row and `n_estimators` columns and should look similar to slide #16 of lecture
        The title of each of the estimator should be iteration number
        Figure 2 should also create a decision surface by combining the individual estimators and should look similar to slide #16 of lecture
        Reference for decision surface: https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html
        This function should return [fig1, fig2]
        """
        color = ["r", "b", "g"]
        Zs = []
        fig1, ax1 = plt.subplots(
            1, len(self.trees), figsize=(5*len(self.trees), 4))

        x_min, x_max = X[0].min(), X[0].max()
        y_min, y_max = X[1].min(), X[1].max()
        x_range = x_max-x_min
        y_range = y_max-y_min

        for i, tree in enumerate(self.trees):
            X_tree, y_tree = self.datas[i]

            xx, yy = np.meshgrid(np.arange(x_min-0.2, x_max+0.2, (x_range)/50),
                                 np.arange(y_min-0.2, y_max+0.2, (y_range)/50))

            ax1[i].set_ylabel("X2")
            ax1[i].set_xlabel("X1")
            Z = tree.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)
            Zs.append(Z)
            cs = ax1[i].contourf(xx, yy, Z, cmap=plt.cm.RdYlBu)
            fig1.colorbar(cs, ax=ax1[i], shrink=0.9)

            for y_label in y.unique():
                idx = y_tree == y_label
                id = list(y_tree.cat.categories).index(y_tree[idx].iloc[0])
                ax1[i].scatter(X_tree.loc[idx, 0], X_tree.loc[idx, 1], c=color[id],
                               cmap=plt.cm.RdYlBu, edgecolor='black', s=30,
                               label="Class: "+str(y_label))
            ax1[i].set_title("Decision Surface Tree: " + str(i+1))
            ax1[i].legend()
        fig1.tight_layout()

        # For Common surface
        fig2, ax2 = plt.subplots(1, 1, figsize=(5, 4))
        Zs = np.array(Zs)
        com_surface, _ = weighted_mode(Zs, np.ones(Zs.shape))
        cs = ax2.contourf(xx, yy, Z, cmap=plt.cm.RdYlBu)
        for y_label in y.unique():
            idx = y == y_label
            id = list(y.cat.categories).index(y[idx].iloc[0])
            ax2.scatter(X.loc[idx, 0], X.loc[idx, 1], c=color[id],
                        cmap=plt.cm.RdYlBu, edgecolor='black', s=30,
                        label="Class: "+str(y_label))
        ax2.set_ylabel("X2")
        ax2.set_xlabel("X1")
        ax2.legend()
        ax2.set_title("Common Decision Surface")
        fig2.colorbar(cs, ax=ax2, shrink=0.9)

        # Saving Figures
        fig1.savefig(os.path.join("figures", "Q4_Fig1.png"))
        fig2.savefig(os.path.join("figures", "Q4_Fig2.png"))
        return fig1, fig2