Investigating the Role of HbA1c Level in Predicting Diabetes Using Different Machine Learning Models
Project Overview
This project investigates how the HbA1c level—a critical biomarker indicating long-term blood sugar control—can be used to predict diabetes status. We use multiple machine learning models to classify whether a patient is diabetic based on clinical data including HbA1c and other relevant features.

The goal is to evaluate and compare the performance of various classification algorithms and identify the best approach for early diabetes prediction.

Dataset Description
Source: Publicly available diabetes datasets at Kaggle

Size:samples with features such as:

HbA1c level

Age

BMI (Body Mass Index)

Blood Pressure

Glucose levels

Insulin levels

smoking history

Other clinical measurements

Target Variable: Diabetes diagnosis (binary classification: diabetic or non-diabetic).

Data Cleaning and Preprocessing
Handled missing values by imputation or removal.

Normalized numerical features to improve model convergence.

Encoded categorical variables if any.

Performed feature scaling using StandardScaler.

Explored correlations between HbA1c and other features.

Exploratory Data Analysis (EDA)
Visualized distributions of HbA1c levels in diabetic vs non-diabetic groups.

Examined correlations between features.

Used boxplots, histograms, and Violin plot to identify key trends.

Found that HbA1c is a strong indicator correlated with diabetes status.

Feature Engineering and Selection
Selected key features based on clinical importance and correlation analysis.

Created interaction terms when relevant.

Focused on HbA1c alongside other metabolic indicators.

Modeling and Evaluation
we trained and compared multiple machine learning classifiers using metrics including Accuracy, Precision, Recall, and F1 Score. Below is a summary of the model performances:

Model	Accuracy	Precision	Recall	F1 Score
Bagging	0.97	0.94	0.70	0.80
Stacking	0.97	0.91	0.67	0.77
Hard Voting	0.9652	0.93	0.64	0.76
Soft Voting	0.9649	0.89	0.67	0.77
XGBoost	0.97	0.96	0.69	0.81
Gradient Boosting	0.97	0.99	0.68	0.81
AdaBoost	0.97	0.97	0.69	0.81
Naive Bayes	0.91	0.46	0.66	0.54
SVM	0.91	0.00	0.00	0.00
Decision Tree	0.95	0.70	0.73	0.72
Logistic Regression	0.96	0.87	0.62	0.73
KNN	0.96	0.90	0.57	0.70
Random Forest	0.97	0.97	0.68	0.80

To visualize these results, the following Python code was used:

python
Copy
Edit
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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

df_melted = df_summary.melt(id_vars='Model', var_name='Metric', value_name='Score')

plt.figure(figsize=(15, 7))
sns.barplot(data=df_melted, x='Model', y='Score', hue='Metric')
plt.title('Comparison of Machine Learning Model Performance Metrics')
plt.xticks(rotation=30, ha='right')
plt.ylim(0, 1)
plt.tight_layout()
plt.show()
Interpretation:
Ensemble models such as Bagging, Stacking, and XGBoost performed best, showing high accuracy and balanced precision and recall. Models like Naive Bayes and SVM had lower predictive performance in this dataset.

Deployment
The best performing model was deployed as a web app (e.g., using Streamlit or Flask) to allow users to input patient data including HbA1c levels and receive a diabetes risk prediction.

Requirements
To run this project locally, install the following packages:

nginx
Copy
Edit
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
streamlit  # or flask, depending on your deployment choice
joblib
Folder Structure
graphql
Copy
Edit
diabetes-predictor/
│
├── app.py                  # Web app code
├── model.pkl               # Trained model file
├── scaler.pkl              # Feature scaler
├── requirements.txt        # Python dependencies
├── README.md               # This file
├── data/                   # Dataset files
└── notebooks/              # Jupyter Notebook
Future Improvements
Incorporate more biomarkers alongside HbA1c.

Use SHAP or LIME for model interpretability.

Add support for batch predictions via CSV upload.

Explore deep learning approaches for enhanced prediction.

Improve web app UI for better user experience.





https://nileuniversity-my.sharepoint.com/:v:/g/personal/m_haitham2296_nu_edu_eg/Efm7UkzXZYdEjh4rbuvxWEcBQuf_Y4EuN4CtJBXxjQUZWA?e=JYxqrz 
[Machine_learning_ppt.pptx](https://github.com/user-attachments/files/20633829/Machine_learning_ppt.pptx)
