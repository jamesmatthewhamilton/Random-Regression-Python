from RandomLasso import random_lasso
import numpy as np
import time
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error
from sklearn import linear_model

def main():
    tests = 1
    start_samples = 50
    start_features = 100
    start_informative = 10
    rmse = np.zeros(tests)
    weights = np.zeros(tests)

    for ii in range(tests):
        x, y, ground_truth = make_regression(n_samples=start_samples, n_features=start_features, #* (ii + 1),
                                             n_informative=start_informative, coef=True)
        print("Ground Truth:\n", ground_truth.T)

        reg = linear_model.LassoCV(normalize=True).fit(x, y)
        print("Pure Lasso Prediction:\n", reg.coef_)
        weights_t = mean_squared_error(reg.coef_, ground_truth)
        print("Pure Lasso RME: ", weights_t)

        start_time = time.time()
        weights = random_lasso(x, y, expected_sampling=40)
        print("--- %s seconds ---" % (time.time() - start_time))
        print("Random Lasso Prediction:\n", weights)
        rmse[ii] = mean_squared_error(weights, ground_truth)
        print("Random Lasso RME:", rmse[ii])

        rmse[ii] = mean_squared_error(weights, ground_truth)

    print(rmse)


if __name__ == "__main__":
    main()
