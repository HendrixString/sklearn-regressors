from sklearn.model_selection import train_test_split as train_test_split__
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator
from sklearn.dummy import DummyRegressor
import pandas as pd
from prepare_data import from_excel_file, select_highly_correlated_columns, target
from eval import evaluate_estimator
from nnls_regressor import NNLS

def train_test_split(df: pd.DataFrame, target_column_name: str=None, 
                    test_size=0.15, random_state=1<<15, **kv):
    """
    create a train test split
    """
    assert target_column_name

    Y = df[target_column_name]
    X = df.drop(target_column_name, axis='columns')

    return train_test_split__(
        X, Y, test_size=test_size, random_state=random_state,
        **kv
        )

def _train_estimator(
        df: pd.DataFrame, target_column_name: str=None,
        model: BaseEstimator=None,
        **kv
        ):
    """
    Train a casual sklearn estimator
    """
    
    assert model
    
    X_train, X_test, y_train, y_test = train_test_split(
        df, target_column_name=target_column_name, **kv
        )

    model.fit(X_train, y_train)

    # print(m.best_estimator_)
    # print(m.best_estimator_.feature_importances_)

    return model, X_test, y_test

def train_estimator(
        model=None,
        **kv
        ):
    """
    Train with sensible defaults
    """

    assert model

    df = from_excel_file()
    df, _ = select_highly_correlated_columns(df)

    return _train_estimator(
        df=df, target_column_name=target,
        model=model, 
        **kv
    )

def train_lasso():
    return train_estimator(
        GridSearchCV(
            estimator=Lasso(),
            cv=5,
            scoring='neg_mean_squared_error',
            verbose=1,
            param_grid={
                'alpha': [float(x) / 1000.0 for x in range(1, 1000, 20)]
            }
        )
    )

def train_NNLS():
    return train_estimator(
        NNLS()
    )

def train_random_forest():
    return train_estimator(
        GridSearchCV(
            estimator=RandomForestRegressor(),
            cv=5,
            scoring='neg_mean_squared_error',
            verbose=1,
            param_grid={
                'n_estimators': range(5, 30, 5),
                'max_depth': range(2, 6),
                'min_samples_split': [float(x) / 10.0 for x in range(1, 6)]
            }
        )
    )

def train_baseline_dummy():
    return train_estimator(
        DummyRegressor(strategy="mean")
    )

if __name__ == "__main__":
    train_evals = [
        train_NNLS, 
        train_lasso,
        train_random_forest,
        train_baseline_dummy
    ]

    for t in train_evals:
        print(f"Train/eval {t.__name__} regressor")
        m, x_test, y_test = t()
        evals = evaluate_estimator(m, x_test, y_test)
        print(evals, '\n')

