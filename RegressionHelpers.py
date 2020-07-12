import numpy as np
from sklearn import linear_model


def ridge_cv(X, y, nfold=5):
    """
    Args:
        X: n x p numpy array, where n is number of instances and p is the number of features.
        y: n x 1 numpy array, the true labels.
        llambda: floating number.
    Return:
        weights: dx1 numpy array, the weights of ridge.
    """
    return None


def lasso_cv(X, y, nfold=5):
    """
    Args:
        X: n x p numpy array, where n is number of samples and p is the number of features.
        y: n x 1 numpy array, the true labels.
        kfold: number of folds for cross validation.
    Return:
        weights: dx1 numpy array, the weights of lasso.
    """
    return None


def elastic_net_cv(X, y, nfold=5):
    """
    Args:
        X: n x p numpy array, where n is number of instances and p is the number of features.
        y: n x 1 numpy array, the true labels.
        kfold: number of folds for cross validation.
    Return:
        weights: dx1 numpy array, the weights of elastic-net.
    """
    return None


def adaptive_lasso(X, y, kfold=5, adaptive_iterations=5, penalty=None, cores=1):
    """
    Args:
        X: n x d numpy array, where n is number of instances and p is the number of features.
        y: n x 1 numpy array, the true labels.
        kfold: number of folds for cross validation.
        penalty: precomputed penalty factor.
        cores: cores for parallel computing.
    Return:
        weights: dx1 numpy array, the weights of adaptive lasso.
    """
    if penalty is None:
        # Initializing penalty with weights from Ridge Regression.
        res_ridge = linear_model.RidgeCV(normalize=False, fit_intercept=False,
                                         kfold=kfold).fit(X, y)
        weights = res_ridge.coef_
    else:
        weights = penalty

    for ii in range(adaptive_iterations):
        print("[Info] Adaptive Lasso on iteration ", ii + 1, " of ", adaptive_iterations, ".", sep='')
        penalty = adaptive_penalty(weights)
        X_new = X / penalty[np.newaxis, :]
        res = linear_model.LassoCV(normalize=False, fit_intercept=False,
                                   kfold=kfold, n_jobs=cores).fit(X_new, y)
        weights = res.coef_ / penalty

    return weights


def adaptive_penalty(weights):
    penalty = 1 / (2 * np.sqrt(np.abs(weights)) + 2.2e-308)
    return penalty
