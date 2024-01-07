
import pandas as pd
from pandas.api.types import is_numeric_dtype, is_string_dtype

target = "Final_Grade"

def prepare_data(df: pd.DataFrame):
    """
    Prepare the data:
    1. remove duplicates (there are none btw)
    2. Learning_Disabilities gymnastics
    3. complete na values with median for numeric dtypes
    4. complete na values with 'unknown' string for string dtypes
    5. one hot encode string columns ( caterogical )
    6. Todo: I can probably remove redundant variables(columns), 
             that are strongly correlated among others.
    """

    # drop duplicates
    df.drop_duplicates(inplace=True)

    # parse "Learning_Disabilities" into columns
    # Learning_Disabilities_{ ADHD / None / ... } 
    # I assume, that missing value indicates no disability
    df_v1 = df["Learning_Disabilities"].fillna('None').str \
        .split(pat = ",", expand=True) \
        .apply(lambda x : x.value_counts(), axis = 1) \
        .add_prefix("Learning_Disabilities_") \
        .fillna(0).astype(int) 
    
    df = pd.concat([df, df_v1], axis=1)
    df = df.drop("Learning_Disabilities", axis=1)

    # Now let's complete other's columns missing values with the following
    # strategy:
    # 1. If column is numeric dtype, use the median to complete na values
    # 2. If column is string dtype, use 'unknown' value to complete na values
    for c in df:
        s = df[c]
        na_val = 'unknown'
        if is_numeric_dtype(s):
            na_val = s.median()
        s.fillna(na_val, inplace=True)    

    # Now, let's one hot encode the categorical columns (non numeric)
    ll = [c for c in df if is_string_dtype(df[c].dtype)]
    df = pd.get_dummies(data=df, columns=ll)

    return df

def from_excel_file(path='./data.xlsx'):
    """
    Load Data from excel file and prepare it
    """
    return prepare_data(pd.read_excel(path))

def select_highly_correlated_columns(df: pd.DataFrame, how_many=10):
    """
    I select the 12 most correlated features against the Final_Grade column
    """
    global target

    corrs = df.corr()[target].abs() \
              .sort_values(ascending=False)[:how_many]
    cols_names = corrs.index.to_list()
    return df[cols_names], corrs    

if __name__ == "__main__":
    df = from_excel_file() 
    df, corrs = select_highly_correlated_columns(df)
    print(corrs)
    print(df.shape)

