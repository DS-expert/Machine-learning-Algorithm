import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import seaborn as sns

# load the data 
df = pd.read_csv("/home/inventor/Datasets/Boston house/boston.csv")

print(df.shape)
print(df.head())
print(df.columns)
print(df.info())
print(df["MEDV"])


# Change the Datypes of Columns
    # AGE -> float to int
    # TAX -> float to int
# Remove the columns
    # ZN
    # CHAS

# Change the Datatypes 
df['AGE'] = df['AGE'].astype(np.int64)
df['TAX'] = df["TAX"].astype(np.int64)

# Remove the Columns
df.drop(columns=["ZN", "CHAS"], axis=1, inplace=True)

print(df.shape)
print(df.head())

# check the corr between features
corr_features = df.corr()
print(corr_features)

# HeatMap
plt.figure(figsize=(10,6))
sns.heatmap(corr_features, annot=True, cmap=sns.color_palette("mako", as_cmap=True), linewidths=0.8)
plt.title("Correlation with features")
plt.show()

# Box Plot
plt.figure(figsize=(10,6))
sns.boxplot(df)
plt.title("BoxPlot of features")
plt.show()

# Identify the outliers using IQR

for i in range(df.shape[1]):

    column_data = df.iloc[:, i]

    q1 = np.percentile(column_data, 25)
    q3 = np.percentile(column_data, 75)
    IQR = q3 - q1

    lower_bound = q1 - (1.5 * IQR)
    upper_bound = q3 + 1.5 * IQR

    outliers = column_data[(column_data < lower_bound) | (column_data > upper_bound)]
    
    perc = len(outliers) / len(column_data) * 100
    print(f"Column {i} outliers = {perc: .2f}%")

X_train, X_test, y_train, y_test = train_test_split(df.drop("MEDV", axis=1), df['MEDV'], test_size=0.2, random_state=42)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

class Multiple_Linear_Regression():

    def __init__(self):
        self.intercept_ = 0
        self.coef_ = 0

    def fit(self, X_train, y_train):
        X_train = np.insert(X_train, 0, 1, axis=1)

        betas = np.linalg.inv(np.dot(X_train.T, X_train)).dot(X_train.T).dot(y_train)

        self.intercept_ = betas[0]
        self.coef_ = betas[1:]
    
    def predict(self, X_test):
        y_pred = np.dot(X_test, self.coef_) + self.intercept_
        return y_pred

mlr = Multiple_Linear_Regression()

mlr.fit(X_train, y_train)

y_pred = mlr.predict(X_test)

def r2_score_custom(y_true, y_pred):
    mean_value = np.mean(y_true)

    SSE = np.sum((y_true - y_pred) ** 2)
    TSS = np.sum((y_true - mean_value) ** 2)

    r2_score = SSE / TSS
    return r2_score

r2_score_c = r2_score_custom(y_true=y_test, y_pred=y_pred)
print(r2_score_c)

from sklearn.linear_model import LinearRegression

lr = LinearRegression()

lr.fit(X_train, y_train)

y_pred_sk = lr.predict(X_test)

from sklearn.metrics import r2_score
r2_score = r2_score(y_true=y_test, y_pred=y_pred_sk)
print(f"R2 Score Custom: {r2_score_c}")
print(f"R2 Score Sckit-learn: {r2_score}")