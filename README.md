# bias_variance_decomposition
* Decompose 0/1 loss and squere loss into bias variance using meny learning methods

## Usage

```
cd bias_variance_decomposition
mkdir data
wget http://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data -O data/pima-indians-diabetes.data
python ./bias_variance_classification.py
```
Output  
```
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

