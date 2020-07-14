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
                                 sample_scale_function,
                                 feature_scale_function,
                                 informative_scale_function,
                                 verbose=False):

    samples = np.full(tests, sample_start)
    features = np.full(tests, feature_start)
    informative = np.full(tests, informative_start)
    hard_coded_tests = np.array(["Least Squares", "Ridge", "Elastic-net", "Lasso", "Adaptive"])

    rme = np.zeros((5, tests))
    f1 = np.zeros((5, tests))
    runtime = np.zeros((5, tests))

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

        cores = (-1 if features[ii] >= 400 else 1)

        if verbose:
            print("Samples:", samples[ii],
                  "| Features:", features[ii],
                  "| Informative:", informative[ii])

        X, y, ground_truth = make_regression(n_samples=samples[ii],
                                             n_features=features[ii],
                                             n_informative=informative[ii],
                                             coef=True)

        # Sorting features by their importance. Most important feature in X[:, 0].
        sorted_indices = np.flip(np.argsort(ground_truth))
        ground_truth = ground_truth[sorted_indices]
        X = X[:, sorted_indices]

        # Testing Least Squares
        start_time = time.time()
        reg = linear_model.LinearRegression().fit(X, y)
        rme[0, ii], f1[0, ii], runtime[0, ii] = \
            bulk_analysis_regression(ground_truth, reg.coef_, "LS", start_time)

        # Testing Ridge
        start_time = time.time()
        reg = linear_model.RidgeCV().fit(X, y)
        rme[1, ii], f1[1, ii], runtime[1, ii] = \
            bulk_analysis_regression(ground_truth, reg.coef_, "Ridge", start_time)

        # Testing Elastic Net
        start_time = time.time()
        reg = linear_model.ElasticNetCV().fit(X, y)
        rme[2, ii], f1[2, ii], runtime[2, ii] = \
            bulk_analysis_regression(ground_truth, reg.coef_, "Elastic", start_time)

        # Testing Lasso
        start_time = time.time()
        reg = linear_model.LassoCV(n_jobs=cores).fit(X, y)
        rme[3, ii], f1[3, ii], runtime[3, ii] = \
            bulk_analysis_regression(ground_truth, reg.coef_, "Lasso", start_time)

        # Testing Adaptive
        start_time = time.time()
        coef = adaptive_lasso(X, y)
        rme[4, ii], f1[4, ii], runtime[4, ii] = \
            bulk_analysis_regression(ground_truth, coef, "Adaptive", start_time)

    sfi = np.array([samples, features, informative])
    return rme, f1, runtime, sfi, hard_coded_tests


def supermassive_regression_plot(title, xlabel, ylabel, footnote, xdata,
                                 ydata, legend, legend_loc="upper right", log=True):

    for ii in range(ydata.shape[0]):
        plt.figure(figsize=(16, 9), dpi=70, facecolor='w', edgecolor='k')

        if log:
            plt.yscale('log')
        colors = ['y', 'b', 'r', 'g', 'm', 'c']
        for jj in range(ydata.shape[1]):
            plt.plot(xdata, ydata[ii, jj], label=legend[jj], color=colors[jj % len(colors)])

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