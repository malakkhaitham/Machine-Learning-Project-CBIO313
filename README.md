# 🧪 Investigating the Role of HbA1c Level in Predicting Diabetes Using Machine Learning

This project explores how **HbA1c levels**, along with other clinical features, can be used to predict diabetes using various **machine learning models**. The final goal is to identify which model performs best and deploy a user-friendly app for prediction.

> 🔍 **Problem Statement**  
> Can we reliably predict whether a person has diabetes using HbA1c levels and other health metrics?

---

## 📂 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Preprocessing](#preprocessing)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Modeling & Evaluation](#modeling--evaluation)
- [Performance Comparison](#performance-comparison)
- [Deployment](#deployment)
- [Installation](#installation)
- [Folder Structure](#folder-structure)
- [Future Improvements](#future-improvements)

---

## 🧠 Overview

This project investigates how well different ML algorithms can classify patients into diabetic or non-diabetic groups, with a focus on the **HbA1c** level — a standard long-term indicator of blood sugar.

We used a set of models ranging from simple logistic regression to ensemble techniques like **Random Forest**, **XGBoost**, **Bagging**, **Voting**, and **Stacking**.

---

## 🗂️ Dataset

- **Source**: Diabetes Dataset Kaggle 
- **Rows**: 100k
- **Features**: 10 clinical features + Target

### ⚙️ Features Used

- Smoking History 
- Glucose  
- Blood Pressure   
- Insulin  
- **HbA1c (approximated via Glucose)**  
- BMI  
-Gender
- Age
- Hypertension
- Heart disease
- **Outcome** (Target: 0 = No Diabetes, 1 = Diabetes)

---

## 🧼 Preprocessing

- Handled missing values using median imputation  
- Scaled features using `StandardScaler`  
- Split data into training and testing sets (80/20)  

---

## 📊 Exploratory Data Analysis

Visualizations included:

- Violin Plot of HbA1c (Glucose) distribution  
- Heatmaps for correlation  
- Boxplots comparing diabetics vs non-diabetics  

> 🔎 We found that **Glucose (HbA1c equivalent)** was one of the strongest predictors of diabetes.

---

## 🤖 Modeling & Evaluation

We trained the following models:

- Logistic Regression  
- Decision Tree  
- K-Nearest Neighbors (KNN)  
- Naive Bayes  
- Support Vector Machine (SVM)  
- Random Forest  
- XGBoost  
- Gradient Boosting  
- AdaBoost  
- Bagging Classifier  
- Stacking Classifier  
- Hard Voting  
- Soft Voting  

All models were evaluated using:

- Accuracy  
- Precision  
- Recall  
- F1 Score  
- ROC-AUC  

---

## 📈 Performance Comparison

Below is the code used to generate the performance bar plot across all models and metrics:

<details>
<summary>🔽 Click to expand code</summary>

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Updated DataFrame with Hard and Soft Voting
df_summary = pd.DataFrame({
    'Model': [
        'Bagging', 'Stacking', 'Hard Voting', 'Soft Voting', 'XGBoost',
        'Gradient Boosting', 'AdaBoost', 'Naive Bayes', 'SVM', 
        'Decision Tree', 'Logistic Regression', 'KNN', 'Random Forest'
    ],
    'Accuracy': [
        0.97, 0.97, 0.9652, 0.9649, 0.97,
        0.97, 0.97, 0.91, 0.91, 0.95, 0.96, 0.96, 0.97
    ],
    'Precision': [
        0.94, 0.91, 0.93, 0.89, 0.96,
        0.99, 0.97, 0.46, 0.00, 0.70, 0.87, 0.90, 0.97
    ],
    'Recall': [
        0.70, 0.67, 0.64, 0.67, 0.69,
        0.68, 0.69, 0.66, 0.00, 0.73, 0.62, 0.57, 0.68
    ],
    'F1 Score': [
        0.80, 0.77, 0.76, 0.77, 0.81,
        0.81, 0.81, 0.54, 0.00, 0.72, 0.73, 0.70, 0.80
    ]
})

# Melt the DataFrame for seaborn plotting
df_melted = df_summary.melt(id_vars='Model', var_name='Metric', value_name='Score')

# Plotting
plt.figure(figsize=(15, 7))
sns.barplot(data=df_melted, x='Model', y='Score', hue='Metric')
plt.title('Comparison of Machine Learning Model Performance Metrics')
plt.xticks(rotation=30, ha='right')
plt.ylim(0, 1)
plt.tight_layout()
plt.show() 

## 🎥 Project Presentation

- 📽️ **Presentation Video**:  
  [Watch the video on OneDrive](https://nileuniversity-my.sharepoint.com/:v:/g/personal/m_haitham2296_nu_edu_eg/Efm7UkzXZYdEjh4rbuvxWEcBQuf_Y4EuN4CtJBXxjQUZWA?e=JYxqrz)  
  This video presents the full walk-through of our machine learning project, including data exploration, modeling, and evaluation results.

- 📊 **Presentation Slides**:  
  [Download the PowerPoint presentation](https://github.com/user-attachments/files/20633829/Machine_learning_ppt.pptx)  
  These slides summarize the project's methodology, model comparisons, and key findings.

