from RandomLasso import random_lasso
import numpy as np
import time
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error


def main():
    tests = 5
    rmse = np.zeros(tests)

    for ii in range(tests):
        x, y, ground_truth = make_regression(n_samples=500, n_features=1000 * (ii + 1),
                                             n_informative=50, coef=True)
        weights = random_lasso(x, y, expected_sampling=80)
        start_time = time.time()
        rmse[ii] = mean_squared_error(weights, ground_truth)
        print("--- %s seconds ---" % (time.time() - start_time))

    print(rmse)


if __name__ == "__main__":
    main()
