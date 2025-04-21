# model.py
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder

def preprocess(df):
    df = df.select_dtypes(include=["number"]).dropna()
    X = df.drop("SalePrice", axis=1)
    y = df["SalePrice"]
    return train_test_split(X, y, test_size=0.2, random_state=42)

def create_model():
    return LinearRegression()

