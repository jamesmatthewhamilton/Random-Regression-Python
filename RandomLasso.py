"""
RandomLasso
Performs regularization using RandomLasso algorithm. Emphases on feature reduction.
High dimensional dataset's prefered, as RandomLasso tends to outperform traditional
regression techniques as the ratio of features to samples increases.

:param x
:param y
:param bootstraps
:param sample
:param alpha
:param box_width
:param nfold
:param verbose
:return: weights
"""

import numpy as np


def random_lasso(x, y, bootstraps=None, expected_sampling=40, alpha=[1, 1], box_width=None,
                nfold=None, verbose=True, verbose_output=True):

    number_of_samples = x.shape[0]
    number_of_features = x.shape[1]
    assert number_of_samples == y.shape[0], \
        "Error: Number of features in x is not equal to the length of y.\n" + \
        "Hint: Input for x had " + str(number_of_samples) + \
        ", and input for y had " + str(y.shape[0]) + " samples."
    assert not np.isnan(x).any(), \
        "Error: The input x array contained at least one NA value."
    assert not np.isnan(y).any(), \
        "Error: The input x array contained at least one NA value."
    assert bootstraps is None or expected_sampling == 40, \
        "Error: Both bootstraps and expected_sampling were set" \
        ", but bootstraps overwrites expected_sampling\n" \
        "Hint: Only one of these two parameters should be set."

    if bootstraps is None:
        bootstraps = np.ceil(number_of_features / number_of_samples) * expected_sampling



    weights = np.zeros((number_of_features, 1))
    return weights
