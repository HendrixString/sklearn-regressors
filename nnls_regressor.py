import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, check_X_y, check_is_fitted
from sklearn.preprocessing import MinMaxScaler
from scipy.optimize import nnls

def to_nd(v):
    if not isinstance(v, np.ndarray):
        return v.to_numpy()
    return v

class NNLS(RegressorMixin, BaseEstimator):
    def __init__(self):
        pass

    def fit(self, X, y, maxiter=1000, use_scaler=True, feature_range=(0, 1)):
        assert y is not None
        
        X, y = check_X_y(X, y)
        self.is_fitted_ = True
        self.n_features_in_ = X.shape[1]

        X, y = to_nd(X), to_nd(y)

        # use an optional scaler
        if use_scaler:
            self._scaler = MinMaxScaler(feature_range=feature_range).fit(X)
            X = self._scaler.transform(X)

        coef_nnls, rnorm = nnls(X, y, maxiter=maxiter)
        self.coef_nnls_ = coef_nnls
        self.rnorm_ = rnorm

        return self

    def predict(self, X):
        check_is_fitted(self, ['is_fitted_', 'coef_nnls_'])

        X = to_nd(X)

        if hasattr(self, '_scaler'):
            X = self._scaler.transform(X)

        pred = X @ self.coef_nnls_

        return pred
        



if __name__ == '__main__':
    # Just a sanity run
    nnls_ = NNLS()

    X, y = np.ones((2, 2)), np.ones(2)

    nnls_.fit(X, y, maxiter=1000, use_scaler=False)
    pred = nnls_.predict(X)

    print('pred: ', pred)
        