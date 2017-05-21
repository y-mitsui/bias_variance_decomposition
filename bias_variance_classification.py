#!/usr/bin/env python
"""
Decompose 0/1 loss into bias variance using meny learning methods in classification

Reference
http://www-bcf.usc.edu/~gareth/research/bv.pdf
"""

import os
import numpy as np
from sklearn.utils import resample
from sklearn.grid_search import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.cross_validation import KFold

from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from classes.ensemble_classifier import EnsembleClassifier
from sklearn.metrics import accuracy_score

def cross_val_score(estimator, sample_X, sample_y):
    est_y = []
    true_y = []
    for train_index, test_index in KFold(sample_X.shape[0], 5):
        X_train2, X_test2, y_train2, y_test2 = sample_X[train_index], sample_X[test_index], sample_y[train_index], sample_y[test_index]
        estimator.fit(X_train2, y_train2)
        est_y.extend(estimator.predict(X_test2).tolist())
        true_y.extend(y_test2.tolist())
    return accuracy_score(est_y, true_y)
    
n_iter = 20

dataset = np.loadtxt(os.path.join("data", "pima-indians-diabetes.data"), delimiter=',')
sample_X = dataset[:, 0:8]
sample_y = dataset[:, 8]

X_train, X_test, y_train, y_test = train_test_split(sample_X, sample_y, test_size=0.2)
print "number of sample ", X_train.shape, X_test.shape

estimators = [
    {"context": EnsembleClassifier(), "tuned_parameters": [], "name": "EnsembleClassifier"},
    {"context": BaggingClassifier(DecisionTreeClassifier(max_depth=14), max_samples=0.9, max_features=0.5, n_estimators=50), "tuned_parameters": [], "name": "Bagging"},
    {"context": RandomForestClassifier(n_estimators=50), "tuned_parameters": [{'max_depth': range(8, 20, 2)}], "name": "random forest"},
    {"context": DecisionTreeClassifier(), "tuned_parameters": [{'max_depth': range(8, 20, 2)}], "name": "decision tree"},
    {"context": LogisticRegression(), "tuned_parameters": [{'C': np.linspace(1e-8, 1e+2, 20)}], "name": "logistic"},
    {"context": KNeighborsClassifier(), "tuned_parameters": [{'n_neighbors': range(1, 10, 1)}], "name": "KNN"},
]


for estimator in estimators:
    print "==========%s========"%(estimator['name'])
    pred = np.zeros((X_test.shape[0], n_iter))
    for i in range(n_iter):
        sample_X, sample_y = resample(X_train, y_train)
        if len( estimator['tuned_parameters']) > 0:
            context = GridSearchCV(estimator['context'], estimator['tuned_parameters'], cv=10, n_jobs=-1, scoring="accuracy")
            context.fit(sample_X, sample_y)
            if i == 0:
                print "grid search results:"
                print "\tbest_params", context.best_params_
                print "\tcross_val_score", context.best_score_ 
        else:
            context = estimator["context"]
            context.fit(sample_X, sample_y)
            if i == 0:
                print "cross_val_score", cross_val_score(context, sample_X, sample_y)
                
        pred[:, i] = context.predict(X_test)
            
    tot_var = 0.
    tot_bias = 0.
    for i in range(X_test.shape[0]):
        target = pred[i, :]
        estimate = 1 if target[target==1].shape[0] > target[target==0].shape[0] else 0
        tot_var += float(target[target!=estimate].shape[0]) / target.shape[0]
        tot_bias += 0 if estimate == y_test[i] else 1
        
    
    print "variance", float(tot_var) / X_test.shape[0]
    print "bias", float(tot_bias) / X_test.shape[0]
            
        
