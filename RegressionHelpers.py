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


def adaptive_lasso(X, y, kfold=5, penalty=None):
    """
    Args:
        X: n x d numpy array, where n is number of instances and p is the number of features.
        y: n x 1 numpy array, the true labels.
        kfold: number of folds for cross validation.
        penalty: precomputed penalty factor.
    Return:
        weights: dx1 numpy array, the weights of adaptive lasso.
    """
    return None
