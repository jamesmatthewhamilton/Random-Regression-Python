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


def RandomLasso(x, y, bootstraps=None, alpha=None, box_width=None,
                nfold=None, verbose=True, verbose_output=True):

    print("Hello Random Lasso")
    weights = np.zeros((x.shape[1]))
    return weights
