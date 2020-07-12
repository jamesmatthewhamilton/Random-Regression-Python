from RandomLasso import random_lasso
import numpy as np
import time
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error
from sklearn import linear_model

def main():
    # Globals for Testing Lasso and Random Lasso.
    tests = 1
    start_samples = 200
    start_features = 500
    start_informative = 3

    rme = np.zeros(tests)
    rme_t = np.zeros(tests)

    for ii in range(tests):
        # Simulating Data
        X, y, ground_truth = make_regression(n_samples=start_samples, n_features=start_features,
                                             n_informative=start_informative, coef=True)

        # Sorting features by their importance. Most important feature in X[:, 0].
        sorted_indices = np.flip(np.argsort(ground_truth))
        ground_truth = ground_truth[sorted_indices]
        X = X[:, sorted_indices]

        print("Ground Truth:\n", ground_truth.T)

        # Testing Lasso
        reg = linear_model.LassoCV().fit(X, y)
        print("Lasso Prediction:\n", reg.coef_)
        rme_t[ii] = mean_squared_error(reg.coef_, ground_truth)
        print("Lasso RME: ", rme_t[ii])

        # Testing and Timing Random Lasso
        start_time = time.time()
        weights = random_lasso(X, y, expected_sampling=40, suppress_warnings=True)
        print("--- %s seconds ---" % (time.time() - start_time))
        print("Random Lasso Prediction:\n", weights)
        rme[ii] = mean_squared_error(weights, ground_truth)
        print("Random Lasso RME:", rme[ii])

    # print("Lasso RMSE:", rmse_t)
    # print("Random Lasso RMSE:", rmse)


if __name__ == "__main__":
    main()
