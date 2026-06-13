# Glucose Spike Prediction Using Machine Learning

## Project Overview

This project was developed for NutriGlyc AI Solutions, a health technology and nutrition analytics company focused on improving diabetes prevention and nutrition management through artificial intelligence and data-driven healthcare solutions.

The objective of the project is to develop a machine learning model capable of predicting glucose spike occurrence using nutrition, health, and lifestyle data. By identifying key risk factors associated with glucose spikes, the solution aims to support early intervention, personalized nutrition recommendations, and improved diabetes management.

## Business Problem
Type 2 Diabetes continues to be a growing public health challenge due to poor dietary habits, sedentary lifestyles, delayed diagnosis, and limited preventive healthcare interventions.
Healthcare providers often face difficulties identifying high-risk individuals early enough to implement effective preventive measures. Traditional assessment methods are often manual, time-consuming, and lack predictive capabilities.

This project addresses these challenges by leveraging machine learning and predictive analytics to support proactive healthcare decision-making.

## Project Objectives

- Predict glucose spike occurrence using health and nutrition data.
- Identify the most influential factors contributing to glucose spikes.
- Compare multiple machine learning algorithms and select the best-performing model.
- Generate actionable insights to support nutrition and lifestyle interventions.
- Demonstrate the potential of AI-driven solutions in preventive healthcare.

## Dataset Description

The dataset contains health, nutrition, and lifestyle-related variables associated with glucose response.

Key features include:

- Age
- Gender
- BMI
- Diabetes Type
- Carbohydrate Intake
- Protein Intake
- Fat Intake
- Fibre Intake
- Sugar Intake
- Glycaemic Index
- Glycaemic Load
- Physical Activity
- Stress Level
I- nsulin Dose
- Smoking Status
- Medication Adherence

Target Variable:

- Glucose Spike (0 = No Spike, 1 = Spike)

## Methodology

The project followed a structured machine learning workflow:

- Data Cleaning and Preparation
- Exploratory Data Analysis (EDA)
- Feature Engineering
- Multicollinearity Assessment
- Feature Selection using Permutation Importance
- Model Development and Comparison
- Hyperparameter Tuning using GridSearchCV
- Final Model Evaluation

## Models Evaluated

The following classification models were compared:

- Logistic Regression
- Random Forest
- XGBoost
- Support Vector Machine (SVM)
- K-Nearest Neighbours (KNN)

Following model comparison and feature selection, Logistic Regression was selected as the final model due to its superior performance and interpretability.

## Final Model Performance

| Metric | Score |
|-----------|-----------|
| Accuracy | 76.0% |
| Precision | 72.1 |
| Recall | 78.6 |
| F1 Score | 75.2 |
| ROC-AUC | 0.848 |

These results indicate that the model can effectively distinguish between glucose spike and non-spike events while maintaining balanced performance across both classes.

## Key Risk Factors Identified

Feature importance analysis identified the following as the most influential predictors of glucose spikes:

- Carbohydrate Intake
- Carbohydrate-to-Fibre Ratio
- Glycaemic Load
- Fibre Intake
- Physical Activity
- Stress Level

These findings highlight the significant role of nutrition and lifestyle factors in glucose management.

## Recommendations

- Monitor carbohydrate intake closely.
- Increase fibre consumption alongside carbohydrate-rich meals.
- Prioritise lower glycaemic load food choices.
- Encourage regular physical activity.
- Promote effective stress management practices.

## Technologies Used
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-Learn
- XGBoost
- Joblib

## Repository Structure

Glucose-Spike-Prediction/
|
├── data/
├── src/
├── models/
├── outputs/
├── README.md
├── requirements.txt
└── .gitignore

## Conclusion

This project demonstrates how machine learning can be applied to healthcare and nutrition data to support early identification of glucose spike risk. The final Logistic Regression model achieved strong predictive performance while remaining interpretable, making it suitable for practical healthcare and nutrition decision-support applications.
