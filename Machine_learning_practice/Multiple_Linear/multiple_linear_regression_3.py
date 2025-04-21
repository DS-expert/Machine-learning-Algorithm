import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

diabetes_data = load_diabetes()
df = pd.DataFrame(diabetes_data.data, columns=diabetes_data.feature_names)
df['target'] = diabetes_data.target

print(df.shape)
print(df.head())
print(df.info())
print(df.describe())
print(df.isnull().sum())
print(df.duplicated().sum())
print(df.corr())

# Plot Heatmap
corr_matrix = df.corr()
plt.figure(figsize=(10,6))
sns.heatmap(corr_matrix, annot=True, cmap=sns.color_palette('mako', as_cmap=True), linewidths=(0.2))
plt.title("Corelation of Features")
plt.show()


plt.figure(figsize=(10,6))
sns.boxplot(df)
plt.title("BoxPlot of features")
plt.show()

# Identify the Outliers

for i in range(df.shape[1]):

    column_data = df.iloc[:, i]

    q1 = np.percentile(column_data, 25)
    q3 = np.percentile(column_data, 75)

    IQR = q3 - q1

    lower_bound = q1 - 1.5 * IQR
    upper_bound = q3 + 1.5 * IQR

    df_cleaned = column_data[(column_data < lower_bound) | (column_data > upper_bound)]

    perc = len(df_cleaned) / len(column_data) * 100
    print(f"Column {i} outliers = {perc:.2f}%")

X_train, X_test, y_train, y_test = train_test_split(df.drop("target", axis=1), df["target"], test_size=0.2, random_state=42)
print(f"\nX_train Shape: {X_train.shape}\nX_test Shape: {X_test.shape}\ny_train Shape: {y_train.shape}\ny_test shape: {y_test.shape}")

class Multiple_Linear_Regression():

    def __init__(self):
        self.intercept_ = None
        self.coef_ = None
    
    def fit(self, X_train, y_train):
        X_train = np.insert(X_train, 0,1, axis=1)
        self.intercept_ = np.array(0)
        self.coef_ = np.ones(X_train.shape[1])

        betas = np.linalg.inv(np.dot(X_train.T, X_train)).dot(X_train.T).dot(y_train)
        self.intercept_ = betas[0]
        self.coef_ = betas[1:]
    
    def predict(self, X_test):
        y_pred = np.dot(X_test, self.coef_) + self.intercept_
        return y_pred
    
mlr = Multiple_Linear_Regression()
mlr.fit(X_train, y_train=y_train)
y_pred = mlr.predict(X_test=X_test)

def r2_score_custom(y_true, y_pred):

    mean_value = np.mean(y_true)
    SSE = np.sum((y_true - y_pred) ** 2)
    TSS = np.sum((y_true - mean_value) ** 2)
    r2_score = SSE / TSS
    return r2_score

r2_score_c = r2_score_custom(y_true=y_test, y_pred=y_pred)
print(f"R2 Score Custom: {r2_score_c}")

# Multiple Linear Regression Scikit-Learn
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_sk = lr.predict(X_test)

r2_score = r2_score(y_true=y_test, y_pred=y_pred_sk)
print(f"R2 Score Custom: {r2_score_c}")
print(f"R2 Score Scikit: {r2_score}")

# Without Outliers DataFrame
print(f"Cleaned Data Shape: {df_cleaned.shape}")
print(f"Cleaned Data: {df_cleaned.head()}")
print(f"Duplicates of Cleaned Data: {df_cleaned.duplicated().sum()}")

