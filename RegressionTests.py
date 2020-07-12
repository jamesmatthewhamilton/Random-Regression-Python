import time
from sklearn.metrics import mean_squared_error


def bulk_analysis_regression(X, y, coef_pred, coef_actual, method, start_time, verbose=True):
    run_time = (time.time() - start_time)  # Since Python passes by reference this is okay.
    rme = mean_squared_error(coef_pred, coef_actual)

    if verbose:
        print(method, "Runtime:", run_time)
        print(method, "RME:", rme)
    return rme, run_time
