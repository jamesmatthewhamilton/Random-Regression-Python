import time
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics import f1_score
from matplotlib import pyplot as plt


def bulk_analysis_regression(coef_true, coef_pred, method, start_time, verbose=False):
    run_time = (time.time() - start_time)  # Since Python passes by reference this is okay.
    rme = mean_squared_error(coef_true, coef_pred)
    f1 = feature_selection_f1_score(coef_true, coef_pred)

    if verbose:
        print("\t", method, "RME:", rme)
        print("\t", method, "F1:", f1)
        print("\t\t", method, ">>>", "%.4f" % run_time, "secs <<<")
        print("\t------------------------------------")
    return rme, f1, run_time


def feature_selection_f1_score(coef_true, coef_pred):
    binary_coef_pred = np.where(coef_pred != 0, 1, 0)
    binary_coef_true = np.where(coef_true != 0, 1, 0)
    return f1_score(binary_coef_true, binary_coef_pred)


def generate_large_plot(title, xlabel, ylabel, footnote, xdata, ydata, legend, legend_loc="upper right", log=True):
    apl = plt.figure(figsize=(16, 9), dpi=75, facecolor='w', edgecolor='k')
    if log:
        plt.yscale('log')
    colors = ['y', 'b', 'r', 'g', 'm', 'c']
    for ii in range(ydata.shape[0]):
        plt.plot(xdata, ydata[ii], label=legend[ii], color=colors[ii % len(colors)])

    plt.legend(loc=legend_loc)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.title(title)
    # plt.text('Note, the ratio of features to informative is constant at 100:5')
    plt.figtext(0.5, 0.05, footnote, ha="center", fontsize=10)

    plt.show()
