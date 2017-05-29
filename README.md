# bias_variance_decomposition
* Decompose 0/1 loss and squere loss into bias variance using many learning methods

# requirements
* xgboost
* hyperopt
* scikit-learn
* numpy

## Usage

```
cd bias_variance_decomposition
mkdir data
wget http://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data -O data/pima-indians-diabetes.data
python ./bias_variance_classification.py
```
Output  
```
==========XGBoost========
grid search results:
	best_params {'reg_alpha': 0.5, 'colsample_bytree': 0.4, 'silent': 1, 'learning_rate': 0.2, 'nthread': -1, 'min_child_weight': 1.0, 'subsample': 0.8, 'reg_lambda': 1.5, 'objective': 'binary:logistic', 'max_depth': 5, 'gamma': 0.2}
	cross_val_score -0.905531120885
variance 0.123701298701
bias 0.25974025974
==========EnsembleClassifier========
cross_val_score 0.889250814332
variance 0.0801948051948
bias 0.246753246753
==========Bagging========
cross_val_score 0.879478827362
variance 0.108441558442
bias 0.227272727273
==========random forest========
grid search results:
	best_params {'max_depth': 18}
	cross_val_score 0.912052117264
variance 0.0931818181818
bias 0.227272727273
==========decision tree========
grid search results:
	best_params {'max_depth': 16}
	cross_val_score 0.895765472313
variance 0.200974025974
bias 0.272727272727
==========logistic========
grid search results:
	best_params {'C': 5.2631579042105265}
	cross_val_score 0.78990228013
variance 0.0418831168831
bias 0.233766233766
==========KNN========
grid search results:
	best_params {'n_neighbors': 1}
	cross_val_score 0.864820846906
variance 0.138636363636
bias 0.318181818182
```

