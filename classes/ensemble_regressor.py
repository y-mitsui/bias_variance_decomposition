from sklearn.ensemble import RandomForestRegressor, BaggingRegressor
from sklearn import tree
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import LassoCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression, Ridge, ElasticNet
import numpy as np

class EnsembleRegressor:
    def __init__(self):
        self.estimators = [
            {"context": BaggingRegressor(tree.DecisionTreeRegressor(max_depth=17), max_samples=0.9, max_features=0.5, n_estimators=50), "tuned_parameters": [], "name": "Bagging"},
            {"context": BaggingRegressor(ElasticNet(alpha=1e-3), max_samples=0.9, max_features=0.5, n_estimators=50), "tuned_parameters": [], "name": "Elastic Bagging"},
            {"context": RandomForestRegressor(n_estimators=50), "tuned_parameters": [{'max_depth': range(5, 20, 3)}], "name": "random forest"},
            {"context": tree.DecisionTreeRegressor(), "tuned_parameters": [{'max_depth': range(1, 20, 2)}], "name": "decision tree"},
            {"context": LassoCV(normalize=True), "tuned_parameters": [{'eps': np.linspace(4e-2, 1, 20)}], "name": "lasso"},
            {"context": KNeighborsRegressor(), "tuned_parameters": [{'n_neighbors': range(1, 50, 10)}], "name": "KNN"},
            {"context": LinearRegression(), "tuned_parameters": [{'normalize': [True, False]}], "name": "Linear"},
            {"context": Ridge(), "tuned_parameters": [{'alpha': np.linspace(2e-2, 1e+2, 40)}], "name": "Ridge"},
            {"context": ElasticNet(), "tuned_parameters": [{'alpha': np.linspace(1e-6, 1e+2, 10), 'l1_ratio':np.linspace(1e-6, 1, 10)}], "name": "ElasticNet"},
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
        return np.average(estimate_values, 0)
