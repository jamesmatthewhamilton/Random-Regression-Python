import numpy as np

def ridge_fit_closed(X, y, llambda):
    """
    Args:
        X: N x D numpy array, where N is number of instances and D is the dimensionality of each instance
        y: N x 1 numpy array, the true labels
        llambda: floating number
    Return:
        weights: Dx1 numpy array, the weights of lasso
    """
    weights = np.dot(np.dot(np.linalg.inv((np.dot(X.T, X)) + (llambda * np.identity(X.shape[1]))), X.T), y)
    return weights


def adaptive_lasso(X, y, penalty):
    """
    Args:
        X: N x D numpy array, where N is number of instances and D is the dimensionality of each instance
        y: N x 1 numpy array, the true labels
        penalty: precomputed penalty factor
    Return:
        weights: Dx1 numpy array, the weights of adaptive lasso
    """
    return None