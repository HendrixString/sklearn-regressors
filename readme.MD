# Intro

## Install
`pip3 install -r requirements.txt`

# Files
## `eda.ipynb`
My own EDA, might be chaotic

## `results.ipynb`
Train / Evaluate / Visualise predictions made by:
- `NNLS`
- `Lasso`
- `Random Forest`
- `DummyRegressor`

## `prepare_data.py`

**`prepare_data()` method**  

Use it to prepare the data:
1. remove duplicates (there are none btw)
2. `Learning_Disabilities` gymnastics into 1 hot encoding
3. complete `na` values with median for numeric dtypes
4. complete `na` values with 'unknown' string for string dtypes
5. one hot encode string columns ( caterogical )
6. Todo: I can probably remove redundant variables(columns), that are strongly correlated among others.
7. Didnt have much time to deep dive


**`select_highly_correlated_columns(df: DataFrame, how_many: Int = 10)` method**  

I select the 12 most correlated features against the `Final_Grade` column

## `train.py`
```python
train_test_split(
df: pd.DataFrame, target_column_name: str=None, test_size=0.15, random_state=1<<15, **kv
)
```
Create a train / test split

```python
train_estimator(
        model=None,
        **kv
        ):
```
Training helper

```python
train_lasso()
train_NNLS()
train_random_forest()
train_baseline_dummy()
```

## `nnls_regressor.py`
`NNLS` Linear regressor with positive coefficients constraint. Implemeted as a `sklearn` `BaseEstimator`

## `vis.py`
some utils for visualizations
