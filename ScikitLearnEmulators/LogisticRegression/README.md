# My Logistic Regression

## Sam Miller

The module `myLogisticRegression` was written as a pure python implementation of 
logistic regression using the fast gradient method. The model can perform 
l2-regularized logistic regression and l2-regularized cross validation of the
logistic regression model. The model currently only supports binomial classification (2-classes)

### Runnning the logistic regression model

To run the model:

* Create an instance of the `LogisticRegression` class
* Fit the model using a standardized data set (see also `myLogisticRegression.standardize_data`)
* Predict or score new data using the `predict` or `score` methods, which utilize the previously fit models.

Note: the y-data should be standardized to -1 or 1 for fitting and scoring purposes. Similarly, the 
	`predict` method returns an array of values of -1 or 1 corresponding to the predicted classes.

```
import myLogisticRegression

clf = myLogisticRegression.LogisticRegression(1.0)
clf.fit(X, y)
clf.score(X, y)
```

### Runnning the cross-validation logistic regression model

Simply by using the `LogisticRegressionCV`, the model will perform k-fold
cross validation on the training set. This is accomplished by dividing
the training set into k approximately equal sized partitions and using each
as a hold-out set for fitting and testing to obtain the regularization penalty
that maximizes the average score across all partitions.

```
import myLogisticRegression

clf = myLogisticRegression.LogisticRegressionCV(cv=5)
clf.fit(X, y)
clf.score(X, y)
```

### Examples

3 Jupyter Notebooks are provided to serve as examples of using the models:

* DemoProblem.ipynb : generates an example data set with 2 predictor variables,
	fits the model using `LogisticRegression` and `LogisticRegressionCV` and
	plots the decision boundary for the two models in the X_1, X_2 plane.
* RealWorldDataset.ipynb : Uses the "Heart.csv" data set which can be downloaded
	here: http://www-bcf.usc.edu/~gareth/ISL/data.html . Fits the model for predicting
	whether or not the patient has Atherosclerotic Heart Disease ("AHD").
* My Model vs Sklearn.ipynb : Uses the `sklearn` data set "breast_cancer" to
	predict whether a tumor is malignant or benign. The models are compared to 
	the `sklearn` `LogisticRegression` and `LogisticRegressionCV` models, demonstrating
	nearly identical performance.