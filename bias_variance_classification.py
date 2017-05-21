#!/usr/bin/env python
# -*- coding: utf-8 -*-
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
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from classes.ensemble_classifier import EnsembleClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import SGDClassifier

def original_cross_val_score(estimator, sample_X, sample_y):
    est_y = []
    true_y = []
    for train_index, test_index in KFold(sample_X.shape[0], 5):
        X_train2, X_test2, y_train2, y_test2 = sample_X[train_index], sample_X[test_index], sample_y[train_index], sample_y[test_index]
        estimator.fit(X_train2, y_train2)
        est_y.extend(estimator.predict(X_test2).tolist())
        true_y.extend(y_test2.tolist())
    return accuracy_score(est_y, true_y)
    
class xgbTuner:
    def __init__(self, estimator, tuned_parameters, cv=5):
        self.estimator = estimator
        self.tuned_parameters = tuned_parameters
        self.cv = cv
        
    def score(self, parameter):
        parameter['max_depth'] = int(parameter['max_depth'])
        self.estimator.set_params(**parameter)
        return -cross_val_score(self.estimator, self.sample_X, self.sample_y, cv=self.cv, scoring="accuracy").mean()
    
    def predict(self, sample_X):
        pred = self.estimator.predict(sample_X)
        return pred
    
    def fit(self, sample_X, sample_y):
        self.sample_X = sample_X
        self.sample_y = sample_y
        
        best_parameter = fmin(self.score, self.tuned_parameters, algo=tpe.suggest, max_evals=200)
        best_parameter.update({"objective":'binary:logistic'})
        best_parameter.update({'silent': 1})
        best_parameter.update({'nthread': -1})
        self.best_score_ = self.score(best_parameter)
        self.best_params_ = best_parameter
        self.estimator.set_params(**best_parameter)
        self.estimator.fit(self.sample_X, self.sample_y)
    
n_iter = 10

dataset = np.loadtxt(os.path.join("data", "pima-indians-diabetes.data"), delimiter=',')
sample_X = dataset[:, 0:8]
sample_y = dataset[:, 8]

X_train, X_test, y_train, y_test = train_test_split(sample_X, sample_y, test_size=0.2)
print "number of sample ", X_train.shape, X_test.shape

xbg_tuning_parameters = {
    'objective': 'binary:logistic',
    'learning_rate': hp.quniform('learning_rate', 0.01, 1, 0.01),
    'gamma': hp.quniform('gamma', 0, 2, 0.1),
    'min_child_weight': hp.quniform('min_child_weight', 0, 10, 1),
    'max_depth': hp.quniform('max_depth', 1, 20, 1),
    'subsample': hp.quniform('subsample', 0.5, 1, 0.1),
    #'colsample_bytree': hp.quniform('colsample_bytree', 0.1, 1, 0.1),
    'colsample_bytree': hp.quniform('colsample_bytree', 0.3, 1, 0.1),
    'reg_lambda': hp.quniform('reg_lambda', 0.0, 1.5, 0.1),
    'reg_alpha': hp.quniform('reg_alpha', 0.0, 1.0, 0.1),
    'nthread': -1,
    'silent': 1,
}
SGD_parameters = [{ 'loss':['perceptron'],
                    'penalty':['l1', 'l2', 'elasticnet'],
                    'alpha': np.linspace(1e-2, 1e+2, 40),
                    'l1_ratio': np.linspace(1e-1, 1, 5),
                    'epsilon':np.linspace(1e-1, 1, 5)
                }]
estimators = [
    {"context": SGDClassifier(n_iter=50), "tuned_parameters": SGD_parameters, "name": "SGD"},
    {"context": XGBClassifier(), "tuned_parameters": xbg_tuning_parameters, "name": "XGBoost", "tuner": xgbTuner(XGBClassifier(), xbg_tuning_parameters)},
    {"context": EnsembleClassifier(), "name": "EnsembleClassifier"},
    {"context": BaggingClassifier(DecisionTreeClassifier(max_depth=14), max_samples=0.9, max_features=0.5, n_estimators=50), "name": "Bagging Tree"},
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
        if estimator.get('tuned_parameters') != None:
            if estimator.get('tuner') != None:
                context = estimator['tuner']
            else:
                context = GridSearchCV(estimator['context'], estimator['tuned_parameters'], cv=5, n_jobs=-1, scoring="accuracy")
            context.fit(sample_X, sample_y)
            if i == 0:
                print "grid search results:"
                print "\tbest_params", context.best_params_
                print "\tcross_val_score", context.best_score_  # resampleによってサンプルに重複が存在するため高く見積もられる
        else:
            context = estimator["context"]
            context.fit(sample_X, sample_y)
            if i == 0:
                print "cross_val_score", original_cross_val_score(context, sample_X, sample_y)
                
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
            
        
