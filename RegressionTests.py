import time
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics import f1_score


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
