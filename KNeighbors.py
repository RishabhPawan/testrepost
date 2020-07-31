# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 21:10:29 2020

@author: risha
"""

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np

iris_dataset = load_iris()

X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'], 
                                                    iris_dataset['target'], 
                                                    random_state=0)

iris_dataframe = pd.DataFrame(X_train, columns=iris_dataset.feature_names)

grr = pd.plotting.scatter_matrix(iris_dataframe, c=y_train, figsize=(15, 15), 
                                 marker='o', hist_kwds={'bins': 20}, 
                                 s=60, alpha=.8)

knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)

x_new = np.array([[5, 2.9, 1, 0.2]])

prediction = knn.predict(x_new)

y_predict = knn.predict(X_test)
knn.score(X_test, y_test)