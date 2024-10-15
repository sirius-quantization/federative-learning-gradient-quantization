import pandas as pd
import math
from typing import List


def add_confidence_interval(df: pd.DataFrame, column: str):
    stats = df.groupby(['model_name', 'epochs', 'batch', 'dataset_name', 'unfreeze', 'gradient_compression', 'slaves_num'])[column].agg(['mean', 'count', 'std'])

    ci95_hi = []
    ci95_lo = []

    for i in stats.index:
        m, c, s = stats.loc[i]
        ci95_hi.append(m + 1.96*s/math.sqrt(c))
        ci95_lo.append(m - 1.96*s/math.sqrt(c))

    df[f'ci95_{column}_hi'] = [ci95_hi] * len(df)
    df[f'ci95_{column}_lo'] = [ci95_lo] * len(df)
    
    
def add_confidence_intervals(df: pd.DataFrame, columns: List[str]):
    for column in columns:
        add_confidence_interval(df, column)
        

def diff(df: pd.DataFrame):
    time_column = "transfer_summary_time_in_memory_sec"
    quantity_column = "transfer_quantity_GB"
    for i in range(len(df) - 1, 0, -1):
        df[time_column][i] = df.iloc[i][time_column] - df.iloc[i - 1][time_column]
        df[quantity_column][i] = df.iloc[i][quantity_column] - df.iloc[i - 1][quantity_column]
