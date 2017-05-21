from sklearn.datasets import load_boston
boston = load_boston()
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LassoCV
from sklearn import tree
from sklearn import svm
from sklearn.cross_validation import KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression, Ridge, ElasticNet
from sklearn.utils import resample
from ensemble_regressor import EnsembleRegressor

def bias_variance(est_y, true_y):
    est_y = np.array(est_y)
    true_y = np.array(true_y)
    
    variance = est_y.var()
    bias = ((est_y - true_y) ** 2).mean()
    return variance, bias, variance / (variance + bias)

def cross_val_score(estimator, parameters, sample_X, sample_y):
    est_y = []
    true_y = []
    for train_index, test_index in KFold(sample_X.shape[0], 5):
        X_train2, X_test2, y_train2, y_test2 = sample_X[train_index], sample_X[test_index], sample_y[train_index], sample_y[test_index]
        estimator.set_params(**parameters)
        estimator.fit(X_train2, y_train2)
        est_y.extend(estimator.predict(X_test2).tolist())
        true_y.extend(y_test2.tolist())
    return bias_variance(est_y, true_y)
    
df = pd.DataFrame(boston.data, columns=boston.feature_names)
df['target'] = boston.target

X_train, X_test, y_train, y_test = train_test_split(boston.data, boston.target, test_size=0.2)
print X_train.shape, X_test.shape

estimators = [
    {"context": EnsembleRegressor(), "tuned_parameters": [], "name": "EnsembleRegressor"},
    {"context": BaggingRegressor(tree.DecisionTreeRegressor(max_depth=12), max_samples=0.9, max_features=0.5, n_estimators=50), "tuned_parameters": [], "name": "Bagging"},
    {"context": BaggingRegressor(ElasticNet(alpha=1e-3), max_samples=0.9, max_features=0.5, n_estimators=50), "tuned_parameters": [], "name": "Elastic Bagging"},
    {"context": RandomForestRegressor(n_estimators=50), "tuned_parameters": [{'max_depth': range(8, 20, 2)}], "name": "random forest"},
    {"context": tree.DecisionTreeRegressor(), "tuned_parameters": [{'max_depth': range(1, 20, 2)}], "name": "decision tree"},
    {"context": LassoCV(normalize=True), "tuned_parameters": [{'eps': np.linspace(4e-2, 1, 20)}], "name": "lasso"},
    {"context": KNeighborsRegressor(), "tuned_parameters": [{'n_neighbors': range(1, 50, 10)}], "name": "KNN"},
    {"context": LinearRegression(), "tuned_parameters": [{'normalize': [True, False]}], "name": "Linear"},
    {"context": Ridge(), "tuned_parameters": [{'alpha': np.linspace(2e-2, 1e+2, 40)}], "name": "Ridge"},
    {"context": ElasticNet(), "tuned_parameters": [{'alpha': np.linspace(1e-6, 1e+2, 10), 'l1_ratio':np.linspace(1e-6, 1, 10)}], "name": "ElasticNet"},
]

for estimator in estimators:
    print "==========%s========"%(estimator['name'])
    if len( estimator['tuned_parameters']) > 0:
        pred = np.zeros((X_test.shape[0], 10))
        for i in range(10):
            sample_X, sample_y = resample(X_train, y_train)
            grid_search = GridSearchCV(estimator['context'], estimator['tuned_parameters'], cv=10, n_jobs=-1, scoring="neg_mean_squared_error")
            grid_search.fit(sample_X, sample_y)
            #print grid_search.best_params_
            #print grid_search.best_score_ 
            #best_params = grid_search.best_params_
            pred[:, i] = grid_search.predict(X_test)
            
        #for i in range(n_repeat):
        #    for j in range(n_repeat):
                #y_error += (y_test[:, j] - pred[:, i]) ** 2
        #y_error /= (n_repeat * n_repeat)
    
        #y_noise = np.var(y_test, axis=1)
        y_bias = (y_test - np.mean(pred, axis=1)) ** 2
        y_var = np.var(pred, axis=1).mean()
        print np.mean(y_bias), y_var
        
        grid_search = GridSearchCV(estimator['context'], estimator['tuned_parameters'], cv=10, n_jobs=-1, scoring="neg_mean_squared_error")
        grid_search.fit(X_train, y_train)
        y_true, y_pred = y_test, grid_search.predict(X_test)
        print mean_squared_error(y_true, y_pred)
    else:
        best_params = {}
        pred = np.zeros((X_test.shape[0], 10))
        for i in range(10):
            sample_X, sample_y = resample(X_train, y_train)
            estimator['context'].fit(sample_X, sample_y)
            pred[:, i] = estimator['context'].predict(X_test)
        
        y_bias = (y_test - np.mean(pred, axis=1)) ** 2
        y_var = np.var(pred, axis=1).mean()
        print np.mean(y_bias), y_var
        
        estimator['context'].fit(X_train, y_train)        
        y_true, y_pred = y_test, estimator['context'].predict(X_test)
        print mean_squared_error(y_true, y_pred)
        
    #print cross_val_score(estimator['context'], best_params, X_train, y_train)


