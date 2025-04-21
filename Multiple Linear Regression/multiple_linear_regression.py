import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression

diabetes_data = load_diabetes()
df = pd.DataFrame(data=diabetes_data.data, columns=diabetes_data.feature_names)
df['target'] = diabetes_data.target

print(df.head())

# Exploration data with Pandas 

print(df.shape)
print(df.info())
print(df.dtypes)
print(df.describe())
print(df.isnull().sum())
print(df.duplicated().sum())
print(df.corr())


# Exploration data with Matplotlib 
df.hist(bins=15, figsize=(10,10))
plt.suptitle("Features of Data")
plt.show()

# boxplot
plt.figure(figsize=(10,6))
sns.boxplot(data=df)
plt.title("BoxPlot of Data")
plt.xticks(rotation=90)
plt.yticks(rotation=90)

# Correlation with Heatmap
plt.figure(figsize=(10,6))
correlation_df = df.corr()
sns.heatmap(correlation_df, annot=True, cmap=sns.color_palette("mako",as_cmap=True), linewidths=0.8)
plt.suptitle("correlation between feature")
plt.show()

# pairplot
# sns.pairplot(df, diag_kind="kde")
# plt.suptitle(f"Paiplot of Features", size=16)
# plt.show()

#   Feature-Target relation

# plt.figure(figsize=(15,10))
# for i, feature in enumerate(df.columns[:-1]):
#     plt.subplot(3, 4, i+2)
#     plt.scatter(df[feature], df['target'], alpha=0.5, color="green")
#     plt.title(f"{feature} vs Target")
#     plt.xlabel(feature)
#     plt.ylabel("Target")

# plt.tight_layout()
# plt.show()

# Single Feature-Target Relation
plt.figure(figsize=(10,6))
plt.scatter(df['bmi'], df['target'], alpha=0.7, color="green")
plt.title(f"BMI VS TARGET")
plt.xlabel("BMI")
plt.ylabel("Target")
plt.tight_layout()
plt.show()

print(df.columns[:-1])

# Custom class of Multiple Linear Regression

class Multiple_Linear_Regression():
    
    def __init__(self):
        self.intercept_ = None
        self.coef_ = None

    def fit(self, X_train, y_train):

        X_train = np.insert(X_train, 0,1, axis=1)

        betas = np.linalg.inv(np.dot(X_train.T, X_train)).dot(X_train.T).dot(y_train)

        self.intercept_ = betas[0]
        self.coef_ = betas[1:]

    def predict(self, X_test):
        y_pred = np.dot(X_test, self.coef_) + self.intercept_
        return y_pred
    
# Split the Data into train and test
X = df.iloc[:, :-1]
y = df["target"]
X_train, X_test, y_train,y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"X shape: {X.shape}, y shape: {y.shape}")
print(f"X_train Shape: {X_train.shape}, X_test Shape: {X_test.shape}, y_train Shape: {y_train.shape}, y_test Shaepe: {y_test.shape}")


mlr = Multiple_Linear_Regression()
mlr.fit(X_train, y_train)

y_pred = mlr.predict(X_test)

def r2_score_c(y_true, y_pred):
    mean_value = np.mean(y_true)

    SSE = np.sum((y_true - y_pred) ** 2)
    TSS = np.sum((y_true - mean_value) ** 2)

    r2_score = SSE / TSS
    
    return r2_score

r2_score_1 = r2_score_c(y_test, y_pred=y_pred)
r2_score_2 = r2_score(y_true=y_test, y_pred=y_pred)
print(f"Custom function R2 Score {r2_score_1}")
print(f"Scikit-learn R2 Score {r2_score_2}")

mlr_sk = LinearRegression()
mlr_sk.fit(X_train, y_train)
y_pred_sk = mlr_sk.predict(X_test)

r2_score_3 = r2_score_c(y_true=y_test, y_pred=y_pred_sk)
r2_score_4 = r2_score(y_true=y_test, y_pred=y_pred_sk)
print(f'Custom Function R2 Score for SLR: {r2_score_3}')
print(f"sckit function R2 Score for SLR: {r2_score_4}")