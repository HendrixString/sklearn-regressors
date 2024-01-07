from sklearn.base import BaseEstimator
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error, mean_absolute_percentage_error
from functools import reduce
from typing import Callable

def evaluate_estimator(e: BaseEstimator, X, y):
    metrics = [
        mean_absolute_error,
        mean_squared_error,
        r2_score
    ]

    pred_y = e.predict(X)

    def reduce_fn(d: dict, m: Callable):
        d[m.__name__] = m(y, pred_y)
        return d

    return reduce(
        reduce_fn,
        metrics,
        dict()
    )
