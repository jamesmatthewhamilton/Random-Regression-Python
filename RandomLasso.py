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
:param sample_size
:param nfold
:param verbose
:return: weights
"""

import numpy as np
from sklearn import linear_model


def random_lasso(x, y, bootstraps=None, expected_sampling=40, alpha=[1, 1], sample_size=None,
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
    if sample_size is None:
        sample_size = number_of_samples

    print(" ------ PART 1 ------ ")
    weights = bootstrap_xy(x, y, bootstraps=bootstraps, sample_size=sample_size)
    importance_measure = np.sum(np.abs(weights), axis=0)
    probability_distribution = importance_measure / np.sum(importance_measure)

    print(" ------ PART 2 ------ ")
    weights = bootstrap_xy(x, y, bootstraps=bootstraps, sample_size=sample_size,
                           probabilities=probability_distribution)
    weights = np.sum(weights, axis=0) / bootstraps

    return weights


def bootstrap_xy(x, y, bootstraps, sample_size, probabilities=None):
    number_of_samples = x.shape[0]
    number_of_features = x.shape[1]

    all_feature_indices = np.arange(number_of_features)
    all_sample_indices = np.arange(number_of_samples)
    bootstrap_matrix = np.zeros((bootstraps, number_of_features))

    for ii in range(bootstraps):
        random_features = np.random.choice(all_feature_indices, sample_size, replace=False, p=probabilities)
        shuffled_samples = np.random.choice(all_sample_indices, sample_size, replace=True)
        new_x = x[:, random_features]  # Valid
        new_x = new_x[shuffled_samples, :]  # Valid
        new_y = y[shuffled_samples]  # Valid

        norm_y = (new_y - np.mean(new_y))  # NV
        norm_x = (new_x - np.mean(new_x, axis=0)) / np.std(new_x, axis=0)  # NV

        reg = linear_model.LassoCV(fit_intercept=False).fit(norm_x, norm_y)  # NV
        bootstrap_matrix[ii, random_features] = reg.coef_  # Valid

    return bootstrap_matrix
