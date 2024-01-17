from sklearn.tree import DecisionTreeRegressor
import numpy as np

class GradientBoostedRegressor:
    def __init__(
        self, base_estimator=None, n_estimators=3, learning_rate=0.1
    ):
        """
        :param base_estimator: The base estimator model instance from which the boosted ensemble is built (e.g., DecisionTree, LinearRegression).
                               If None, then the base estimator is DecisionTreeRegressor(max_depth=1).
                               You can pass the object of the estimator class
        :param n_estimators: The maximum number of estimators at which boosting is terminated. In case of perfect fit, the learning procedure may be stopped early.
        :param learning_rate: The learning rate shrinks the contribution of each tree by `learning_rate`.
        """
        self.base_estimator = (
            DecisionTreeRegressor(max_depth=1)
            if base_estimator is None
            else base_estimator
        )
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.estimators_ = []

    def fit(self, X, y):
        """
        Function to train and construct the GradientBoostedRegressor
        Inputs:
        X: pd.DataFrame with rows as samples and columns as features (shape of X is N X P) where N is the number of samples and P is the number of columns.
        y: pd.Series with rows corresponding to output variable (shape of Y is N)
        """
        self.estimators_ = []
        y_pred = np.zeros_like(y)
        for i in range(self.n_estimators):
            residual = y - y_pred
            estimator = self.base_estimator.fit(X, residual)
            self.estimators_.append(estimator)
            y_pred += self.learning_rate * estimator.predict(X)

    def predict(self, X):
        """
        Input:
        X: pd.DataFrame with rows as samples and columns as features
        Output:
        y: pd.Series with rows corresponding to output variable. THe output variable in a row is the prediction for sample in corresponding row in X.
        """
        y_pred = np.zeros(len(X))
        for estimator in self.estimators_:
            y_pred += self.learning_rate * estimator.predict(X)
        return y_pred

