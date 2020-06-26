"""
RandomLasso
Performs regularization using RandomLasso algorithm. Emphases on feature reduction.
High dimensional dataset's prefered, as RandomLasso tends to outperform traditional
regression techniques as the ratio of features to samples increases.

:param x
:param y
:param bootstraps
:param alpha
:param box_width
:param nfold
:param verbose
:return: weights
"""

import numpy as np


def random_lasso(x, y, bootstraps=None, alpha=None, box_width=None,
                nfold=None, verbose=True, verbose_output=True):
    number_of_samples = x.shape[0]
    number_of_features = x.shape[1]
    assert number_of_samples == y.shape[0], \
        "Error: Number of features in x is not equal to the length of y."

    weights = np.zeros((number_of_features, 1))
    return weights
