import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


def heatmap(df: pd.DataFrame, figsize=(20,10)):
    sns.heatmap(df.corr(numeric_only=True), annot=True)
    plt.rcParams['figure.figsize'] = figsize
    plt.show()