import time
import numpy as np
from RegressionHelpers import adaptive_lasso
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import f1_score
from matplotlib import pyplot as plt
from tqdm import tqdm  # Progress Bar
from sklearn import linear_model


def supermassive_regression_test(tests,
                                 sample_start,
                                 feature_start,
                                 informative_start,
                                 noise_start=0.0,
                                 effective_rank_start=None,
                                 sample_scale_function=None,
                                 feature_scale_function=None,
                                 informative_scale_function=None,
                                 noise_scale_function=None,
                                 effective_rank_scale_function=None,
                                 hard_coded_tests=None,
                                 verbose=False):

    assert (isinstance(sample_start, int))
    assert (isinstance(feature_start, int))
    assert (isinstance(informative_start, int))
    if noise_start is not None:
        assert(isinstance(noise_start, float))
    if effective_rank_start is not None:
        assert(isinstance(effective_rank_start, int))

    samples = np.full(tests, sample_start)
    features = np.full(tests, feature_start)
    informative = np.full(tests, informative_start)
    noise = np.full(tests, noise_start)
    effective_rank = np.full(tests, effective_rank_start)

    if hard_coded_tests is None:
        hard_coded_tests = np.array(["Least Squares", "Ridge", "Elastic-net L1=0.2",  "Elastic-net L1=0.5",
                                     "Elastic-net L1=0.8", "Lasso", "Adaptive"])

    if sample_scale_function is None:
        def sample_scale_function(x, y, z, ii):
            return x

    if feature_scale_function is None:
        def sample_scale_function(x, y, z, ii):
            return y

    if informative_scale_function is None:
        def sample_scale_function(x, y, z, ii):
            return z

    n_methods = len(hard_coded_tests)

    rme = np.zeros((n_methods, tests))
    f1 = np.zeros((n_methods, tests))
    runtime = np.zeros((n_methods, tests))

    for ii in tqdm(range(tests)):
        if verbose:
            print("------------ Test", ii + 1, "of", tests, "------------")

        samples[ii] = np.round(sample_scale_function(samples[ii],
                                                     features[ii],
                                                     informative[ii], ii))

        features[ii] = np.round(feature_scale_function(samples[ii],
                                                       features[ii],
                                                       informative[ii], ii))

        informative[ii] = np.round(informative_scale_function(samples[ii],
                                                              features[ii],
                                                              informative[ii], ii))
        if noise_scale_function is not None:
            noise[ii] = noise_scale_function(noise[ii], ii)

        if effective_rank_scale_function is not None:
            effective_rank[ii] = effective_rank_scale_function(effective_rank[ii], ii)

        cores = (-1 if features[ii] >= 400 else 1)

        if verbose:
            print("Samples:", samples[ii],
                  "| Features:", features[ii],
                  "| Informative:", informative[ii],
                  "| Noise:", noise[ii],
                  "| Effective Rank", effective_rank[ii])

        X, y, ground_truth = make_regression(n_samples=samples[ii],
                                             n_features=features[ii],
                                             n_informative=informative[ii],
                                             noise=noise[ii],
                                             effective_rank=effective_rank[ii],
                                             coef=True)

        # Sorting features by their importance. Most important feature in X[:, 0].
        sorted_indices = np.flip(np.argsort(ground_truth))
        ground_truth = ground_truth[sorted_indices]
        X = X[:, sorted_indices]

        for jj in range(n_methods):

            # Testing Least Squares
            if (np.any(hard_coded_tests[jj] == "Least Squares")):
                start_time = time.time()
                reg = linear_model.LinearRegression().fit(X, y)
                rme[jj, ii], f1[jj, ii], runtime[jj, ii] = \
                    bulk_analysis_regression(ground_truth, reg.coef_, hard_coded_tests[jj], start_time)

            # Testing Ridge
            if (np.any(hard_coded_tests[jj] == "Ridge")):
                start_time = time.time()
                reg = linear_model.RidgeCV().fit(X, y)
                rme[jj, ii], f1[jj, ii], runtime[jj, ii] = \
                    bulk_analysis_regression(ground_truth, reg.coef_, hard_coded_tests[jj], start_time)

            # Testing Elastic Net L1=0.2
            if (np.any(hard_coded_tests[jj] == "Elastic-net L1=0.2")):
                start_time = time.time()
                reg = linear_model.ElasticNetCV(l1_ratio=0.2).fit(X, y)
                rme[jj, ii], f1[jj, ii], runtime[jj, ii] = \
                    bulk_analysis_regression(ground_truth, reg.coef_, hard_coded_tests[jj], start_time)

            # Testing Elastic Net L1=0.5
            if (np.any(hard_coded_tests[jj] == "Elastic-net L1=0.5")):
                start_time = time.time()
                reg = linear_model.ElasticNetCV(l1_ratio=0.5).fit(X, y)
                rme[jj, ii], f1[jj, ii], runtime[jj, ii] = \
                    bulk_analysis_regression(ground_truth, reg.coef_, hard_coded_tests[jj], start_time)

            # Testing Elastic Net L1=0.8
            if (np.any(hard_coded_tests[jj] == "Elastic-net L1=0.8")):
                start_time = time.time()
                reg = linear_model.ElasticNetCV(l1_ratio=0.8).fit(X, y)
                rme[jj, ii], f1[jj, ii], runtime[jj, ii] = \
                    bulk_analysis_regression(ground_truth, reg.coef_, hard_coded_tests[jj], start_time)

            # Testing Lasso
            if (np.any(hard_coded_tests[jj] == "Lasso")):
                start_time = time.time()
                reg = linear_model.LassoCV(n_jobs=cores).fit(X, y)
                rme[jj, ii], f1[jj, ii], runtime[jj, ii] = \
                    bulk_analysis_regression(ground_truth, reg.coef_, hard_coded_tests[jj], start_time)

            # Testing Adaptive
            if (np.any(hard_coded_tests[jj] == "Adaptive")):
                start_time = time.time()
                coef = adaptive_lasso(X, y)
                rme[jj, ii], f1[jj, ii], runtime[jj, ii] = \
                    bulk_analysis_regression(ground_truth, coef, hard_coded_tests[jj], start_time)

    sfinr = np.array([samples, features, informative, noise, effective_rank], dtype=object)
    return rme, f1, runtime, sfinr, hard_coded_tests


def supermassive_regression_plot(title, xlabel, ylabel, footnote, xdata,
                                 ydata, legend, legend_loc="upper right",
                                 colors=None, lines=None, log=True):

    for ii in range(ydata.shape[0]):
        plt.figure(figsize=(16, 9), dpi=70, facecolor='w', edgecolor='k')

        if log:
            plt.yscale('log')
        if colors is None:
            colors = ['y', 'b', 'r', 'g', 'm', 'c']
        if lines is None:
            lines = ['-']

        for jj in range(ydata.shape[1]):
            plt.plot(xdata, ydata[ii, jj], lines[jj % len(lines)], label=legend[jj], color=colors[jj % len(colors)])

        plt.legend(loc=legend_loc[ii])
        plt.ylabel(ylabel[ii])
        plt.xlabel(xlabel)
        plt.title(title)
        plt.figtext(0.5, 0.05, footnote, ha="center", fontsize=10)

        plt.show()


def bulk_analysis_regression(coef_true, coef_pred, method, start_time=None, verbose=False):

    if start_time is not None:
        runtime = (time.time() - start_time)  # Since Python passes by reference this is okay.
    else:
        runtime = None
    rme = mean_squared_error(coef_true, coef_pred)
    f1 = feature_selection_f1_score(coef_true, coef_pred)

    if verbose:
        print("\t", method, "RME:", rme)
        print("\t", method, "F1:", f1)
        if start_time is not None:
            print("\t\t", method, ">>>", "%.4f" % runtime, "secs <<<")
        print("\t------------------------------------")

    if start_time is not None:
        return rme, f1, runtime
    else:
        return rme, f1


def feature_selection_f1_score(coef_true, coef_pred):
    binary_coef_pred = np.where(coef_pred != 0, 1, 0)
    binary_coef_true = np.where(coef_true != 0, 1, 0)
    return f1_score(binary_coef_true, binary_coef_pred)


def root_mean_sq_error(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def predict(xtest, weight):
    N = xtest.shape[0]
    prediction = np.zeros((N, 1))
    prediction[:, 0] = np.sum(xtest * weight.T, axis=1)
    np.shape(prediction.shape)
    return prediction


def cross_validation(X, y, method, kfold=10, l1_ratio=0.5):
    N = X.shape[0]
    rmse_error = np.zeros(kfold)
    all_indices = np.arange(N)
    bag_of_indices = all_indices
    amount_to_sample = np.int(np.floor((1 / kfold) * N))
    for ii in range(kfold):
        test_index = np.random.choice(bag_of_indices, amount_to_sample, replace=False)
        for jj in range(len(test_index)):
            bag_of_indices = np.delete(bag_of_indices, np.where(bag_of_indices == test_index[jj]), axis=0)
        train_index = np.delete(all_indices, test_index)
        x_train = X[train_index, :]
        y_train = y[train_index]
        if method == "ols":
            reg = linear_model.LinearRegression().fit(x_train, y_train)
            coef = reg.coef_
        if method == "ridge":
            reg = linear_model.RidgeCV().fit(x_train, y_train)
            coef = reg.coef_
        if method == "elastic":
            reg = linear_model.ElasticNetCV(n_jobs=-1, l1_ratio=l1_ratio).fit(x_train, y_train)
            coef = reg.coef_
        if method == "lasso":
            reg = linear_model.LassoCV(n_jobs=-1).fit(x_train, y_train)
            coef = reg.coef_
        if method == "adaptive":
            coef = adaptive_lasso(x_train, y_train)
        x_test = X[test_index, :]
        y_test = y[test_index]
        prediction = predict(x_test, coef)
        rmse_error[ii] = root_mean_sq_error(prediction, y_test)
    return np.mean(rmse_error)

