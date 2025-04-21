import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

diabetes_data = load_diabetes()
df = pd.DataFrame(diabetes_data.data, columns=diabetes_data.feature_names)
df['target'] = diabetes_data.target

print(df.shape)
print(df.head())
print(df.info())
print(df.isnull().sum())
print(df.duplicated().sum())

# identify the outliers
for i in range(df.shape[1]):
    column_data = df.iloc[:, i]

    q1 = np.percentile(column_data, 25)
    q3 = np.percentile(column_data, 75)

    IQR = q3 - q1

    lower_bound = q1 - 1.5 * IQR
    upper_bound = q3 + 1.5 * IQR

    outliers = column_data[(column_data < lower_bound) | (column_data > upper_bound)]

    perc = len(outliers) / len(df) * 100
    print(f'Column {i} outliers = {perc: .2f}%')

class Batch_Gradient_Descent():

    def __init__(self, learning_rate, epochs):
        self.coef_ = None
        self.intercept_ = None
        self.lr = learning_rate
        self.epochs = epochs
    
    def fit(self, X_train, y_train):
        X_train = np.insert(X_train, 0,1, axis=1)
        self.intercept_ = 0
        self.coef_ = np.ones(X_train.shape[1])

        for i in range(self.epochs):
            y_hat = self.intercept_ + np.dot(X_train, self.coef_)
            intercept_slope = -2 * np.mean((y_train - y_hat))
            self.intercept_ = self.intercept_ + (self.lr * intercept_slope)

            coef_slope = -2 * np.mean((y_train - y_hat) * X_train.shape[0])
            self.coef_ = self.coef_ + (self.lr * coef_slope)
    
    def predict(self, X_test):
        X_test = np.insert(X_test, 0,1, axis=1)
        y_pred = np.dot(X_test, self.coef_) + self.intercept_
        return y_pred
    
bgd = Batch_Gradient_Descent(learning_rate=0.01, epochs=50)

X_train, X_test, y_train, y_test = train_test_split(df.drop("target", axis=1), df['target'], test_size=0.2, random_state=42)

bgd.fit(X_train, y_train)

y_pred = bgd.predict(X_test)

def r2_score_custom(y_true, y_pred):
    mean_value = np.mean(y_true)

    SSE = np.sum((y_true - y_pred) ** 2)
    TSS = np.sum((y_true - mean_value) ** 2)

    r2_score = SSE / TSS
    return r2_score

r2_score_c = r2_score_custom(y_true=y_test, y_pred=y_pred)
print(f"\nR2 Score Custom: {r2_score_c}")

lr = LinearRegression()

lr.fit(X_train, y_train)

y_pred_sk = lr.predict(X_test)

r2_score_sk = r2_score(y_true=y_test, y_pred=y_pred_sk)
r2_score_sk_c = r2_score_custom(y_true=y_test, y_pred=y_pred_sk)
print(f"R2 Score Scikit-learn: {r2_score_sk}")
print(f"Custom R2 Score for Sckit-learn: {r2_score_sk_c}")

