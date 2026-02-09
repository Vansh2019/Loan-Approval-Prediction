# ==============================
# Loan Approval Prediction
# Logistic Regression (FINAL)
# ==============================

# Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, roc_auc_score

# ==============================
# Load Dataset
# ==============================

df = pd.read_csv("C:/Users/hp/Downloads/loandata.csv")

print("First 5 Rows:")
print(df.head())

print("\nDataset Info:")
print(df.info())

print("\nStatistical Summary:")
print(df.describe())

# ==============================
# Data Preprocessing
# ==============================

df.drop("Loan_ID", axis=1, inplace=True)

df['Gender'].fillna(df['Gender'].mode()[0], inplace=True)
df['Married'].fillna(df['Married'].mode()[0], inplace=True)
df['Dependents'].fillna(df['Dependents'].mode()[0], inplace=True)
df['Self_Employed'].fillna(df['Self_Employed'].mode()[0], inplace=True)
df['LoanAmount'].fillna(df['LoanAmount'].median(), inplace=True)
df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mode()[0], inplace=True)
df['Credit_History'].fillna(df['Credit_History'].mode()[0], inplace=True)

df['Dependents'] = df['Dependents'].replace('3+', 3).astype(int)

# ==============================
# Feature Engineering
# ==============================

df['TotalIncome'] = df['ApplicantIncome'] + df['CoapplicantIncome']
df['LoanIncomeRatio'] = df['LoanAmount'] / df['TotalIncome']

# ==============================
# Encode Categorical Variables
# ==============================

le = LabelEncoder()
categorical_cols = ['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area', 'Loan_Status']

for col in categorical_cols:
    df[col] = le.fit_transform(df[col])

# ==============================
# Extra Graphs (EDA Visuals)
# ==============================

# 1. Loan Status Distribution
plt.figure(figsize=(6,4))
sns.countplot(x='Loan_Status', data=df)
plt.title("Loan Approval Distribution")
plt.show()

# 2. Income Distribution
plt.figure(figsize=(7,4))
plt.hist(df['TotalIncome'], bins=20)
plt.title("Total Income Distribution")
plt.xlabel("Income")
plt.ylabel("Frequency")
plt.show()

# 3. Loan Amount vs Loan Status
plt.figure(figsize=(6,4))
sns.boxplot(x='Loan_Status', y='LoanAmount', data=df)
plt.title("Loan Amount vs Loan Status")
plt.show()

# 4. Correlation Heatmap
plt.figure(figsize=(10,6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.show()

# ==============================
# Split Data
# ==============================

X = df.drop('Loan_Status', axis=1)
y = df['Loan_Status']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ==============================
# Feature Scaling
# ==============================

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ==============================
# Train Model
# ==============================

model = LogisticRegression()
model.fit(X_train, y_train)

# ==============================
# Predictions
# ==============================

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# ==============================
# Evaluation
# ==============================

print("\nAccuracy:", round(accuracy_score(y_test, y_pred)*100, 2), "%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# ==============================
# ROC Curve
# ==============================

fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = roc_auc_score(y_test, y_prob)

plt.figure()
plt.plot(fpr, tpr, label="ROC Curve (AUC = %0.2f)" % roc_auc)
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()

# ==============================
# Feature Importance
# ==============================

importance = model.coef_[0]
features = X.columns

plt.figure(figsize=(10,5))
sns.barplot(x=importance, y=features)
plt.title("Feature Importance")
plt.show()
