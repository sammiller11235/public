"""
The following models were created as a pure python
implementation similar to the sklearn.linear_model

Author: Samuel Miller
Date:   6/7/2019
"""

import numpy as np
from scipy import linalg
from matplotlib import pyplot as plt


def split_data_set(x, y, split = 0.2, seed = 0):
    """
    Splits the data into a training and test set.
    
    Parameters
    ----------
    x : np.ndarray
        2D array of floats
    
    y : np.ndarray
        1D array of outputs
    
    split : float, optional
        A value between 0 and 1 that represents the fraction
        of data used in the test set
        
    seed : int, optional
        A seed value used to generate the random sample
    
    Returns
    -------
    4 lists: x_test, y_test, x_train, y_train
        by splitting the data set with the split ratio.
    """
    np.random.seed(seed)
    
    # Generate a set of random integers of size n
    n = len(y)
    shuffled_idx = np.random.permutation(list(range(n)))
    
    # Split the randomized indices
    test_size = int(np.ceil(split*n))
    test_indices = shuffled_idx[:test_size]
    train_indices = shuffled_idx[test_size:]
    
    # Split the data based on the randomized indices
    x_test = np.array([x[i] for i in test_indices])
    y_test = np.array([y[i] for i in test_indices])
    x_train = np.array([x[i] for i in train_indices])
    y_train = np.array([y[i] for i in train_indices])
    
    return x_test, y_test, x_train, y_train

def split_k_folds(x, y, k_folds, seed = 0):
    """
    Splits the data into k groups with randomization
    
    Parameters
    ----------
    x : np.ndarray
        2D array of floats
    
    y : np.ndarray
        1D array of outputs
    
    k_folds : int
        Number of folds for the split
        
    seed : int, optional
        A seed value used to generate the random sample
    
    Returns
    -------
    x_k : list 
        k groups of x values
    y_k : list 
        k groups of y values
    """
    np.random.seed(seed)
    
    # Generate a set of random integers of size n
    n = len(y)
    shuffled_idx = np.random.permutation(list(range(n)))
    
    k_fold_groups = np.array_split(shuffled_idx, k_folds)
    # Split the randomized indices
    x_k = []
    y_k = []
    for i, k_i in enumerate(k_fold_groups):
        x_k.append(x[k_i])
        y_k.append(y[k_i])
    return np.array(x_k), np.array(y_k)

def standardize_data(x_train, x_test):
    """
    Standardizes the data by using the training set.
    
    Parameters
    ----------
    x_train : np.ndarray
        2D array of floats, the training data set
        
    x_test : np.ndarray
        2D array of floats, the test data set
    """
    d = len(x_train[0])
    
    for i in range(d):
        # Compute mean and standard deviation of training set
        x_j_bar = np.mean(x_train[:,i])
        stand_dev = np.std(x_train[:,i])

        # Perform in-place standardization
        x_train[:,i] = (x_train[:,i] - x_j_bar)/stand_dev
        x_test[:,i]  = (x_test[:,i] - x_j_bar)/stand_dev
        
    return x_train, x_test


class RegressionModel:
    """
    The RegressionModel class implements the fast gradient descent method.
    It is inherited by other classes which must implement the objective function (f_beta)
    and the gradient (computegrad). 
    """
    def __init__(self, lambda_, tol=1.e-5, max_iter=1000):
        """
        lambda_ : float
            shrinkage penalty for regularized models.
        """
        self.lambda_ = lambda_
        self.tol = tol
        self.max_iter = max_iter
        self.coeff_list = None
        self.coef_ = None
        self.C_ = None

    
    def backtracking(self, x, y, beta, eta, iter_max = 100):
        """
        Returns an improved step size eta based on the backtracking line-search method.
        
        Parameters
        ----------
        x : numpy.array
            matrix i by j, predictors
        y : numpy.array
            matrix i, response values {-1, 1}
        beta : numpy.array
            matrix j
        eta : float
            step size
        """
        alpha=0.5
        gamma=0.8
     
        grad_f = self.computegrad(x, y, beta)
        norm_grad_f = np.linalg.norm(grad_f)
        iteration_exit = False
        i = 0
        while (iteration_exit is False) and i < iter_max:
            lhs = self.f_beta(x, y, beta - (eta*grad_f))
            rhs = self.f_beta(x, y, beta) - alpha*eta*norm_grad_f**2
            if lhs < rhs:
                iteration_exit = True
            else:
                eta = eta*gamma
                i += 1
        return eta
    
    
    def fastgradalgo(self, x, y, beta_init, eta0=1):
        """
        Performs the accelerated gradient descent method for logisitic regression
        
        Parameters
        ----------
        x : numpy.array
            matrix i by j, predictors
        y : numpy.array
            matrix i, response values {-1, 1}
        beta_init : numpy.array
            matrix j, initial estimate of beta matrix
        eta0 : float
            initial step size
        epsilon : float
            target accuracy, compared to ||gradF(beta)||
        mat_iter : int
            maximum number of allowed iterations
            
        Returns
        -------
        beta_vals : (list of list of model coefficients for each iteration
           List of calculated beta values at each iteration
        """
        n,d = x.shape
        beta = beta_init
        theta = beta_init
        beta_vals = [beta]
        i = 0
        eta = eta0
        grad_theta = self.computegrad(x, y, theta)
        grad_beta = self.computegrad(x, y, beta)
        
        while np.linalg.norm(grad_beta) > self.tol and i < self.max_iter:
            eta = self.backtracking(x, y ,beta, eta)
            beta_tp1 = theta - eta * grad_theta
            theta_tp1 = beta_tp1 + i/(i+3) * (beta_tp1-beta)
            theta = theta_tp1
            beta_vals.append(beta_tp1)
            
            grad_theta = self.computegrad(x, y, theta)
            grad_beta = self.computegrad(x, y, beta)
            beta = beta_tp1
            i += 1 
            
        return np.array(beta_vals)
    
    def estimate_step_size(self, x, y):
        """
        Calculates an estimate of the initial step size based on the Lipschitz smoothness
        """
        n, d = x.shape
        return 1./(linalg.eigh(1./len(y)*x.T.dot(x), 
                                     eigvals=(d-1, d-1), 
                                     eigvals_only=True)[0]+self.lambda_)

    def plot_learning(self, x, y):
        """
        Returns a plot of the model learning over the iterations
        """
        objective_iter = [self.f_beta(x, y, coef) for coef in self.coeff_list]
        fig = plt.figure()
        plt.plot(range(len(self.coeff_list)), objective_iter)
        plt.xlabel('Iteration')
        plt.ylabel('Objective Function')
        
        return fig


class LogisticRegression(RegressionModel):
    """
    Model for l2-regularized logistic regression using the fast gradient descent method.
    """
        
    def f_beta(self, x, y, beta):
        """
        Returns the objective function F(beta)
        
        Parameters
        ----------
        x : numpy.array
            matrix i by j, predictors
        y : numpy.array
            matrix i, response values {-1, 1}
        beta : numpy.array
            matrix j
        """
        yxbeta = y*x.dot(beta)
        expTerm = np.exp(-yxbeta)
        logTerm = np.sum(np.log(1 + expTerm))
        term1 = (1/y.shape[0])*(logTerm)
        term2 = self.lambda_*np.linalg.norm(beta)**2
        return term1 + term2
    	
    def computegrad(self, x, y, beta):
        """
        Returns the gradient of the objective function
        
        Parameters
        ----------
        x : numpy.array
            matrix i by j, predictors
        y : numpy.array
            matrix i, response values {-1, 1}
        beta : numpy.array
            matrix j
        """
        n, d = x.shape
        xbeta = x.dot(beta) 
        p_i = xbeta * -y
        frac = (np.exp(p_i)/(1 + np.exp(p_i)))
        matrixP1 = np.diagflat(frac.tolist())
        return -(1/n)*(x.T.dot(matrixP1).dot(y)) + 2*self.lambda_*beta

        
    def fit(self, x, y):
        """
        Fits the model for the parameters given
        
        Notes
        -----
        x should be standardized before passing to this function
        to ensure robust fitting of parameters
        
        y should be an array of values of -1 and 1
        """
        n, d = x.shape
        self.C_ = 1/(2*n*self.lambda_) # Stored for consistency with sklearn.LogisticRegression
        
        beta_init = np.random.normal(size=d)
        #beta_init = np.zeros(d)
        eta0 = self.estimate_step_size(x, y)
        coeff_list = self.fastgradalgo(x, y, beta_init, eta0 = eta0)
        
        self.coeff_list = coeff_list
        self.coef_ = coeff_list[-1]
        
    def predict(self, x):
        """
        Returns an array of predicted parameters based on the model fit.
        
        Notes
        -----
        x should be standardized before passing to this function
        to ensure robust fitting of parameters
        """
        if type(self.coef_) == type(None):
            raise ValueError("The model has not yet been fit.")
        
        rhs = np.exp(x.dot(self.coef_))
        p = rhs/(1+rhs)
        return (p > 0.5)*2 - 1
    
    def score(self, x, y):
        """
        Returns a score for the x and y data provided.
        
        Notes
        -----
        x should be standardized before passing to this function
        to ensure robust fitting of parameters
        
        y should be an array of values of -1 and 1
        """
        n, d = x.shape
        y_pred = self.predict(x)
        return (1/n)*np.sum(y_pred == y)
    
class LogisticRegressionCV(LogisticRegression):
    """
    Model for l2-regularized logistic regression with cross-validation
    using the fast gradient descent method.
    """
        
    def __init__(self, cv=5, lambda_list=None, tol=1.e-5, max_iter=1000):
        """
        cv : int
            number of fold used in cross-valdiation
        
        lambda_list : list of floats
            shrinkage penalties to try
            
        tol : float
            tolerance applied to gradient descent algorithm
            
        max_iter : int
            maximum number of iterations for gradient descent algorithm
        """
        self.cv = cv
        self.lambda_list = lambda_list or np.logspace(-4, 4, 10)
        self.tol = tol
        self.max_iter = max_iter
        
        self.cv_model = None   # optimal model determined by cross-validation
        self.lambda_ = None    # optimal lambda determined by cross-validation
        self.coef_ = None      # final coefficients using optimal lambda
        self.C_ = None         # C_ parameter using optimal lambda
        
    def fit(self, x, y, plot_scores = True):
        """
        Fits the model for the parameters given by
        performing k-fold cross validation by partitioning the data 
        into k-folds, fitting the model for each regularization penalty
        on the holdout sets.
        
        Returns
        -------
        lambda_ : float
            optimal regularization penalty
        """
        n, d = x.shape
        x_k, y_k = split_k_folds(x, y, self.cv)
        
        self.score_lam = np.zeros(len(self.lambda_list))
        for m, lam in enumerate(self.lambda_list):
            model = LogisticRegression(lam)
            
            score = np.zeros(self.cv)
            for i in range(self.cv):
                x_train = np.concatenate(np.delete(x_k, i, axis=0))
                y_train = np.concatenate(np.delete(y_k, i, axis=0))
                x_test = x_k[i]
                y_test = y_k[i]
                 
                model.fit(x_train, y_train)
                score[i] = model.score(x_test, y_test)
            self.score_lam[m] = np.mean(score)
             
        self.lambda_ = self.lambda_list[np.argmax(self.score_lam)]
        self.C_ = 1/(2*n*self.lambda_)
        self.cv_model = LogisticRegression(self.lambda_)
        self.cv_model.fit(x, y)
        self.coef_ = self.cv_model.coef_
    
    def plot_cv_scores(self):
        """ 
        Plots the cross-validation score vs. regularization penalty lambda) 
        """
        fig = plt.figure()
        plt.semilogx(self.lambda_list, self.score_lam, marker='o', linestyle='-', color='black')
        plt.xlabel(r'$\lambda$')
        plt.ylabel('CV-Score')
        return fig
            
    def predict(self, x):
        """
        Returns an array of predicted parameters based on the model fit.
        
        Notes
        -----
        x should be standardized before passing to this function
        to ensure robust fitting of parameters
        """
        return self.cv_model.predict(x)
    
    def score(self, x, y):
        """
        Returns a score for the x and y data provided.
        
        Notes
        -----
        x should be standardized before passing to this function
        to ensure robust fitting of parameters
        
        y should be an array of values of -1 and 1
        """
        return self.cv_model.score(x, y)
        