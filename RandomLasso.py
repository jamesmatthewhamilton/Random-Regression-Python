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


def random_lasso(X, y, bootstraps=None, expected_sampling=40, alpha=[1, 1], sample_size=None,
                 nfold=None, verbose=True, verbose_output=True):
    number_of_samples = X.shape[0]
    number_of_features = X.shape[1]

    assert number_of_samples == y.shape[0], \
        "Error: Number of features in x is not equal to the length of y.\n" + \
        "Hint: Input for x had " + str(number_of_samples) + \
        ", and input for y had " + str(y.shape[0]) + " samples."
    assert not np.isnan(X).any(), \
        "Error: The input x array contained at least one NA value."
    assert not np.isnan(y).any(), \
        "Error: The input x array contained at least one NA value."
    assert bootstraps is None or expected_sampling == 40, \
        "Error: Both bootstraps and expected_sampling were set" \
        ", but bootstraps overwrites expected_sampling\n" \
        "Hint: Only one of these two parameters should be set."

    # Calculating bootstraps when user does not specify.
    if bootstraps is None:
        bootstraps = int(np.ceil(number_of_features / number_of_samples) * expected_sampling)
    # Setting sample size when user does not specify.
    if sample_size is None:
        sample_size = number_of_samples

    print(" ------ PART 1 ------ ")
    bootstrap_matrix = bootstrap_Xy(X, y, bootstraps=bootstraps, sample_size=sample_size)
    weights = np.sum(np.abs(bootstrap_matrix), axis=0)
    probability_distribution = weights / np.sum(weights)

    print(" ------ PART 2 ------ ")
    # Using the results of the weights from Part 1 in our random sampling.
    weights = bootstrap_Xy(X, y, bootstraps=bootstraps, sample_size=sample_size,
                           probabilities=probability_distribution)
    weights = np.sum(weights, axis=0) / bootstraps

    return weights


def bootstrap_Xy(X, y, bootstraps, sample_size, probabilities=None):
    number_of_samples = X.shape[0]
    number_of_features = X.shape[1]

    all_feature_indices = np.arange(number_of_features)  # 0, 1, 2 ... #features
    all_sample_indices = np.arange(number_of_samples)  # 0, 1, 2, ... #samples
    bootstrap_matrix = np.zeros((bootstraps, number_of_features))  # (bootstraps x features)

    for ii in range(bootstraps):
        # Randomly sampling sample_size number of feature indices.
        random_features = np.random.choice(all_feature_indices, size=sample_size, replace=False, p=probabilities)
        # Randomly shuffling and duplicating sample_size number of sample indices.
        shuffled_samples = np.random.choice(all_sample_indices, size=sample_size, replace=True)

        # Generating a new X and y based on the above random indices.
        new_x = X[:, random_features]  # Valid
        new_x = new_x[shuffled_samples, :]  # Valid
        new_y = y[shuffled_samples]  # Valid

        # Standardizing the new X and y.
        norm_y = (new_y - np.mean(new_y))  # NV
        std_x = np.std(new_x, axis=0)  # NV
        norm_x = (new_x - np.mean(new_x, axis=0)) / std_x  # NV

        # Running some flavor of regression. Uses k-fold cross validation to tune hyper-parameter.
        reg = linear_model.LassoCV(fit_intercept=False).fit(norm_x, norm_y)  # BUG HERE: "ConvergenceWarning"
        # Adding to large bootstrap matrix. Will get the sum of each column later.
        bootstrap_matrix[ii, random_features] = reg.coef_ / std_x  # Valid

    return bootstrap_matrix
