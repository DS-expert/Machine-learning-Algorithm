# data_loader.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as masno

def load_data(path):
    df = pd.read_csv(path)
    df.drop(columns=["Alley", "Pool QC", "Fence", "Misc Feature"], axis=1, inplace=True)
    return df

def show_missing(df):
    masno.bar(df)
    masno.heatmap(df)
    plt.show()

