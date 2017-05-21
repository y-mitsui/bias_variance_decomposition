import numpy as np
from sklearn.grid_search import GridSearchCV

from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

class EnsembleClassifier:
    def __init__(self):
        self.estimators = [
            {"context": BaggingClassifier(DecisionTreeClassifier(max_depth=14), max_samples=0.9, max_features=0.5, n_estimators=50), "tuned_parameters": [], "name": "Bagging"},
            {"context": RandomForestClassifier(n_estimators=50), "tuned_parameters": [{'max_depth': range(8, 20, 2)}], "name": "random forest"},
            {"context": DecisionTreeClassifier(), "tuned_parameters": [{'max_depth': range(8, 20, 2)}], "name": "decision tree"},
            {"context": LogisticRegression(), "tuned_parameters": [{'C': np.linspace(1e-8, 1e+2, 20)}], "name": "logistic"},
            {"context": KNeighborsClassifier(), "tuned_parameters": [{'n_neighbors': range(1, 10, 1)}], "name": "KNN"},
        ]
    
    def set_params(self):
        pass
        
    def fit(self, sample_X, sample_y):
        self.fitted_estimators = []
        for estimator in self.estimators:
            if len( estimator['tuned_parameters']) > 0:
                grid_search = GridSearchCV(estimator['context'], estimator['tuned_parameters'], cv=10, n_jobs=-1, scoring="neg_mean_squared_error")
                grid_search.fit(sample_X, sample_y)
                self.fitted_estimators.append(grid_search)
            else:
                best_params = {}
                estimator['context'].fit(sample_X, sample_y)
                self.fitted_estimators.append(estimator['context'])

    def predict(self, sample_X):
        estimate_values = []
        for estimator in self.fitted_estimators:
            estimate_values.append(estimator.predict(sample_X))
        estimate_values = np.array(estimate_values)
        
        pred_y = []
        for i in range(sample_X.shape[0]):
            target = estimate_values[:, i]
            estimate = 1 if target[target==1].shape[0] > target[target==0].shape[0] else 0
            pred_y.append(estimate)
            
        return np.array(pred_y)
