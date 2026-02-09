# 🏦 Loan Approval Prediction - Machine Learning Project

## 📌 Overview
This project builds a **Loan Approval Prediction system** using Machine Learning to predict whether a bank loan application will be **Approved or Rejected** based on applicant financial and demographic data.

The project demonstrates **Finance domain understanding, Statistical Analysis, Data Preprocessing, Feature Engineering, and Classification Modeling** using Logistic Regression.

---

## 🎯 Objective
To develop a predictive model that helps financial institutions evaluate loan eligibility based on applicant information such as income, credit history, loan amount, and employment details.

---

## 🧠 Features
- Complete **Machine Learning Pipeline**
- Data Cleaning & Missing Value Handling
- Feature Engineering (Total Income, Loan-Income Ratio)
- Exploratory Data Analysis (EDA)
- Statistical Analysis
- Logistic Regression Classification Model
- Model Evaluation using Accuracy, Precision, Recall, F1-score, ROC-AUC
- Data Visualization & Insights

---

## 📊 Dataset
The dataset contains loan applicant information such as:

- Gender
- Marital Status
- Dependents
- Education
- Employment Status
- Applicant Income
- Coapplicant Income
- Loan Amount
- Loan Term
- Credit History
- Property Area
- Loan Status (Target Variable)

Sample dataset included: **`loan_data.csv`**

---

## 🛠️ Technologies Used
- **Python**
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Machine Learning (Logistic Regression)
- Statistical Analysis

---

## ⚙️ Machine Learning Workflow

1. Data Loading
2. Data Cleaning & Missing Value Treatment
3. Feature Engineering
4. Encoding Categorical Variables
5. Feature Scaling
6. Train-Test Split
7. Model Training (Logistic Regression)
8. Model Evaluation
9. Visualization & Insights

---

## 📈 Model Performance
- Accuracy: ~80% – 88%
- Evaluated using:
  - Confusion Matrix
  - Precision
  - Recall
  - F1 Score
  - ROC-AUC Curve

---

## 📊 Visualizations Included
- Loan Approval Distribution (Count Plot)
- Income Distribution (Histogram)
- Loan Amount vs Loan Status (Box Plot)
- Feature Correlation Heatmap
- ROC Curve
- Feature Importance Graph

These visualizations help understand feature impact and model performance.

---

## 🚀 How to Run the Project

### Step 1: Install Dependencies
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
Step 2: Run the Model

###Make sure loan_data.csv is in the same folder as the Python script.

python loan_prediction.py

