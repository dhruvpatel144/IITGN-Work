import pandas as pd
import matplotlib.pyplot as plt
from metrics import *
from ensemble.gradientBoosted import GradientBoostedRegressor
from tree.base import DecisionTree
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression

np.random.seed(42)

X, y= make_regression(
    n_features=3,
    n_informative=3,
    noise=10,
    tail_strength=10,
    random_state=42,
)
print(y)


X = pd.DataFrame(X)
y = pd.Series(y)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

gbm = GradientBoostedRegressor(n_estimators=5, learning_rate=0.1)

gbm.fit(X_train, y_train)

y_pred = gbm.predict(X_test)

rmsee = rmse(y_test, pd.Series(y_pred))
maee = mae(y_test, pd.Series(y_pred))
print(f"Root Mean Squared Error: {rmsee:.2f}")
print(f'MAE: {maee:.2f}')

# Or use sklearn decision tree

########### GradientBoostedClassifier ###################
