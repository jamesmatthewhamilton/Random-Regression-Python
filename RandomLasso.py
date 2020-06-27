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
        bootstraps = int(np.ceil(number_of_features / number_of_samples) * expected_sampling)

    all_feature_indices = np.arange(number_of_features)
    all_sample_indices = np.arange(number_of_samples)

    for ii in range(bootstraps):
        random_features = np.random.choice(all_feature_indices, number_of_samples, replace=False)
        shuffled_samples = np.random.choice(all_sample_indices, number_of_samples, replace=True)
        new_x = x[:, random_features]
        new_x = new_x[shuffled_samples, :]
        new_y = y[shuffled_samples]

    print(new_x.shape)
    print(new_y.shape)
    print("index features \n", random_features)
    print("index samples \n", shuffled_samples)
    print("random features \n", new_x)
    print("shuffled samples \n", new_y)

    weights = np.zeros((number_of_features, 1))
    return weights
