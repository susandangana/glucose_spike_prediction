#-----------------------------------------------------------------------------------------------------
# OVERVIEW
#-----------------------------------------------------------------------------------------------------
# NutriGlyc AI Solutions is a health technology company focused on improving diabetes prevention and 
# nutrition management through artificial intelligence and data analytics. This project aims to 
# develop a machine learning-based solution that predicts the risk of Type 2 Diabetes and 
# glucose spikes using health, lifestyle, and nutrition data.
# The solution leverages predictive analytics to support early risk identification, 
# personalized nutrition recommendations, and data-driven healthcare decision-making. 
# By combining healthcare data processing, machine learning, and interactive visualizations, 
# the project seeks to enhance preventive care, improve patient outcomes, and 
# support healthcare professionals in delivering more effective and personalized interventions. 
# Ultimately, the project demonstrates how AI-driven healthcare solutions can 
# contribute to better diabetes management and long-term health outcomes.
#-------------------------------------------------------------------------------------------------------

# %%
# LIBRARIES
#---------------------------------------------------------------------------------------------------------------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix, classification_report
from sklearn.inspection import permutation_importance
import joblib


# %%
# DATA LOADING AND INSPECTATION
#---------------------------------------------------------------------------------------------------------------------------------------------------------
df = pd.read_csv('../data/Glucose_spike_Dataset - Sheet1.csv')

# View First rows
print(df.head())

# Dataset information
print(df.info())

# Summary statistics of numerical columns
summary_stats = df.describe()
summary_stats.to_excel('summary_statistics.xlsx')
print(summary_stats)

#--------------------------------------------------------------------------------------------
# Insight from Initial Data Inspection
# An initial review of the dataset and its summary statistics reveals that 
# the dataset contains 5,150 observations across demographic, dietary, 
# lifestyle, medication, and glucose-related variables. 
# Patients have an average age of 48 years and an average BMI of 27.0. 
# Meals contain an average of 122.5g of carbohydrates and 
# have a mean glycemic load of 73.6. 
# Pre-meal glucose levels average 110.7 mg/dL, 
# increasing to 144.7 mg/dL after meals, resulting in an average glucose change of 34 mg/dL. 
# Medication adherence is relatively high at 81%. 
# The target variable, glucose_spike, is fairly balanced, 
# with 46.5% of observations classified as spikes and 53.5% as non-spikes, 
# providing a suitable distribution for predictive modelling.
#--------------------------------------------------------------------------------------------

#%%
# DATA CLEANING
#--------------------------------------------------------------------------------------------
# Missing Value
missing = df.isnull().sum()
print(missing)

# Before deciding on what to do with the missing values we inspect distribution of the features affected
stats = df[['fiber_intake', 'insulin_dose', 'stress_level', 'sleep_hours']].describe()
print(stats)

df[['fiber_intake', 'insulin_dose', 'stress_level', 'sleep_hours']].hist(figsize=(10,8))
plt.show()

# Replace missing values in these affected features with the median
for col in [
    'fiber_intake', 
    'insulin_dose', 
    'stress_level', 
    'sleep_hours']:

    df[col] = df[col].fillna(df[col].median())

# Confirm fill
print(df.isnull().sum())

# Check for duplicate entries
print(df.duplicated().sum())

# View the duplicated rows
print(df[df.duplicated()])

# Drop these duplicated rows
df = df.drop_duplicates()

# Verify duplicates removal
print(df.duplicated().sum())

# Drop patient_id as it is not relevant to our model
df = df.drop('patient_id', axis=1)

# Confirm drop
#print(df.head())

# View the now cleaned data information
df.info()

#------------------------------------------------------------------------------------------------------------
# During data cleaning, missing values were identified in four features: 
# fiber_intake (309), insulin_dose (311), stress_level (315), and sleep_hours (310). 
# To determine the most appropriate imputation method, 
# the summary statistics of these variables were examined, 
# with particular attention paid to their mean and median values. 
# Given the presence of outliers, missing values were imputed using the median, 
# as it is more robust to extreme observations. A further inspection revealed 
# 150 duplicate records, which were removed from the dataset. 
# Additionally, the patient_id feature was dropped as it does not provide predictive value for the model. 
# Following these cleaning steps, the dataset contained 5,000 observations and 27 features, 
# with no missing values or duplicate records remaining.
# ---------------------------------------------------------------------------------------------------------

# %%
# EXPLORATORY DATA ANALYSIS
#----------------------------------------------------------------------------------------------------------

# Univariate Analysis (Categorical Features) - Check distribution of features and column
print(df['gender'].value_counts())

print(df['diabetes_type'].value_counts())

print(df['meal_time'].value_counts())

print(df['smoking_status'].value_counts())

print(df['alcohol_consumption'].value_counts())

print(df['medication_adherence'].value_counts())

# Check the balance of the target variable
print(df['glucose_spike'].value_counts(normalize=True))

# Visualize this balance
sns.countplot(x='glucose_spike', data=df)
plt.show()
#---------------------------------------------------------------------------------
# The univariate analysis of the categorical variables shows that the dataset is relatively balanced across 
# most categorical variables. Female patients slightly outnumber males, 
# while Type 2 diabetes accounts for the majority of observations (85.4%). 
# Meal times are evenly distributed across breakfast, lunch, dinner, and 
# snack categories. Most patients reported no smoking or alcohol consumption, and 
# medication adherence was high, with approximately 81% of patients 
# adhering to their prescribed treatment. As said before, 
# the target variable glucose_spike has a fairly balanced distribution.
#-----------------------------------------------------------------------------------

# Univariate Analysis (Numerical Features) - Check distribution of features
# get list of numerical columns in df
numeric_cols = [col for col in df.columns if col not in ['gender', 
                                                         'diabetes_type',
                                                         'meal_time',
                                                         'smoking_status',
                                                         'alcohol_consumption',
                                                         'medication_adherence',
                                                         'glucose_spike']]
print(numeric_cols)

# Number of numeric features
print(len(numeric_cols))

# View the distribution of the numerical features
fig, ax = plt.subplots(nrows=5, ncols=4, figsize=(30, 30))
ax = ax.flatten()

for idx, col in enumerate(numeric_cols):
    sns.histplot(ax=ax[idx], data=df[col])
    ax[idx].set_title(f"Histogram plot of {col}")
    
plt.show()

# Further check for outliers
fig, ax = plt.subplots(nrows=10, ncols=2, figsize=(50, 100))
ax = ax.flatten()

for idx, col in enumerate(numeric_cols):
    sns.boxplot(y=df[col], ax=ax[idx])
    ax[idx].set_title(f"Boxplot of {col}")

plt.tight_layout()
plt.show()
#------------------------------------------------------------------------------------------------------------------------------------------------
# The histogram and boxplot analyses of the numerical variables, 
# provide consistent evidence regarding the distribution and 
# outlier characteristics of the numerical features. 
# While variables such as BMI, protein intake, fat intake, portion size, 
# water intake, pre-meal glucose, post-meal glucose, and glucose change 
# exhibit approximately bell-shaped distributions, features including glycemic load, 
# insulin dose, carb_fiber_ratio, and meal_risk_score show noticeable positive skewness. 
# The boxplots confirm these observations by highlighting the presence of outliers, 
# particularly in the skewed variables, while features such as age, 
# stress level, sleep hours, and glycemic index display relatively few 
# extreme values. As the identified outliers fall within plausible clinical and behavioural ranges 
# rather than representing obvious data entry errors, 
# they would be retained to preserve potentially important information that may influence glucose spike prediction.
# Rather than clip or remove these extreme values in these features, 
# I will engineer new features of logarithmic transformation of these features to reduce skewness, while preserving information,
# and compare these with the original features during modelling using 
# fearure importance or model performance.
#-----------------------------------------------------------------------------------------------------------------------------------------

# Bivariate Anallysis 
# Numerical Features vs Target

# Does higher carbonhydrate intake influence the occurrence of glucose spikes?
print(df.groupby('glucose_spike')['carb_intake'].mean())
# Visualise
plt.figure(figsize=(4, 6))

sns.boxplot(x='glucose_spike', y='carb_intake', data=df)
plt.title('Carbonhydrate Intake vs Glucose Spike')
plt.show()

# Does sugar intake contribute to glucose spike events?
print(df.groupby('glucose_spike')['sugar_intake'].mean())
# Visualise
plt.figure(figsize=(6, 4))
sns.boxplot(x='glucose_spike', y='sugar_intake', data=df)
plt.title('Sugar Intake vs Glucose Spike')
plt.show()

# Is glycemic index associated with glucose spike occurence?
print(df.groupby('glucose_spike')['glycemic_index'].mean())
# Visualise
plt.figure(figsize=(6,4))
sns.boxplot(x='glucose_spike', y='glycemic_index', data=df)
plt.title('Glycemic Index vs Glucose Spike')
plt.show()

# Does glycemic load increase the likelihood of glucose spikes?
print(df.groupby('glucose_spike')['glycemic_load'].mean())
# Visualise
plt.figure(figsize=(6, 4))
sns.boxplot(x='glucose_spike', y='glycemic_load', data=df)
plt.title('Glycemic Load vs Glucose Spike')
plt.show()

# Is Meal Risk score a useful indicator of glucose spike risk?
print(df.groupby('glucose_spike')['meal_risk_score'].mean())
# Visualise
plt.figure(figsize=(6, 4))
sns.boxplot(x='glucose_spike', y='meal_risk_score', data=df)
plt.title('Meal Risk Score vs Glucose Spike')
plt.show()

# Does pre-meal glucose level affect post-meal spike occurrence?
print(df.groupby('glucose_spike')['pre_meal_glucose']
      .agg(['mean', 'median'])
      )
# Visualise
sns.boxplot(
    x='glucose_spike',
    y='pre_meal_glucose',
    data=df
)

# Is physical activity associated with lower glucose spike rate?
print(df.groupby('glucose_spike')['physical_activity'].mean())
# Visualise
plt.figure(figsize=(6, 4))
sns.boxplot(x='glucose_spike', y='physical_activity', data=df)
plt.title('Physical Activity vs Glucose Spike')
plt.show()

# Categorical Features vs Target
# Does glucose spike occurrence vary across different meal times?
pd.crosstab(
    df['meal_time'],
    df['glucose_spike'],
    normalize='index'
)*100
# Visualise
sns.countplot(
    x='meal_time', 
    hue='glucose_spike', 
    data=df
)
plt.title('Meal Time vs Glucose Spike')
plt.show()

# Does the type of diabetes influence glucose spike frequency?
pd.crosstab(
    df['diabetes_type'],
    df['glucose_spike'],
    normalize='index'
)
# Visualise
sns.countplot(
    x='diabetes_type',
    hue='glucose_spike',
    data=df
)
plt.title('Diabetes Type vs Glucose Spike')
plt.show()

# Does medication adherence impact the likelihood of glucose spike?
pd.crosstab(
    df['medication_adherence'],
    df['glucose_spike'],
    normalize='index'
)
# Visualise
sns.countplot(
    x='medication_adherence',
    hue='glucose_spike',
    data=df
)
plt.title('Medication Adherence vs Glucose Spike')
plt.show()

#------------------------------------------------------------------------------------------------------
# The bivariate analysis reveals that several meal-related features are 
# strongly associated with glucose spike occurrence. Patients who experienced glucose spikes 
# had noticeably higher average carbohydrate intake, glycemic load, and meal risk scores 
# compared to those who did not experience spikes, suggesting that meal composition plays 
# a significant role in post-meal glucose responses. 
# In contrast, sugar intake, glycemic index, and pre-meal glucose levels 
# showed very little difference between the two groups, indicating a weaker relationship 
# with glucose spike occurrence. Additionally, patients with glucose spikes 
# tended to have slightly lower levels of physical activity, 
# suggesting a potential protective effect of exercise. Among the categorical variables, 
# glucose spike rates appeared relatively similar across different meal times, diabetes types, and 
# medication adherence groups, indicating that these factors may have a less pronounced influence 
# on spike occurrence within this dataset. Overall, carbohydrate-related measures, particularly 
# glycemic load and meal risk score, emerged as the strongest indicators of glucose spike risk.

# Multivariate Analysis
# Check for multicollinearity by computing the correlation between the numerical features
df.head()
numeric_cor = df[numeric_cols].corr() 
print(numeric_cor)
#Visualise
plt.figure(figsize=(20, 10))
sns.heatmap(numeric_cor, annot=True, fmt='.2f')
plt.title('Heatmap of Numerical Features and Target')
plt.show()
#--------------------------------------------------------------------------------------------------------------------------------
# The correlation heatmap indicates that most features exhibit weak pairwise correlations, 
# suggesting a generally low risk of multicollinearity across the dataset. 
# However, several strong relationships are observed among engineered and glucose-related variables. 
# Notably, carb_fiber_ratio and meal_risk_score show a very strong positive correlation (0.92), 
# while post_meal_glucose and glucose_change are also highly correlated (0.91). 
# Additionally, carb_intake is strongly correlated with glycemic_load (0.87) and 
# moderately correlated with glucose_change (0.70) and post_meal_glucose (0.63). 
# These strong correlations are expected, as some of these variables are derived from or 
# closely related to one another. Overall, the analysis suggests that 
# multicollinearity is limited to a small number of engineered and outcome-related features, 
# while the majority of predictors remain relatively independent. 
# The identified highly correlated features will be considered during feature selection to minimise redundancy and 
# reduce the potential impact of multicollinearity during model development.
#--------------------------------------------------------------------------------------------------------------------------------

# %%
# FEATURE ENGINEERING
#--------------------------------------------------------------------------------------------------------------------------------
# New columns holding Logorithmic transformed features identified to be highly skewed (containing outlier) identified during EDA
df['log_carb_fiber_ratio'] = np.log1p(df['carb_fiber_ratio'])

df['log_meal_risk_score'] = np.log1p(df['meal_risk_score'])

df['log_glycemic_load'] = np.log1p(df['glycemic_load'])

df['log_insulin_dose'] = np.log1p(df['insulin_dose'])

skewed_n_log_cols  = ['carb_fiber_ratio', 
                      'log_carb_fiber_ratio', 
                      'meal_risk_score', 
                      'log_meal_risk_score', 
                      'glycemic_load', 
                      'log_glycemic_load', 
                      'insulin_dose', 
                      'log_insulin_dose'
                      ]

# View the distribution of these transformed features compared to the original features
fig, ax = plt.subplots(nrows=4, ncols=2, figsize=(20, 20))
ax = ax.flatten()

for idx, col in enumerate(skewed_n_log_cols):
    sns.histplot(ax=ax[idx], data=df[col])
    ax[idx].set_title(f"Histogram of {col}")

plt.show()
#--------------------------------------------------------------------------------------------------------------------------------------------------
# As part of the feature engineering process, logarithmic transformations were applied to carb_fiber_ratio, meal_risk_score, glycemic_load, 
# and insulin_dose to reduce the impact of skewness and extreme values identified during exploratory data analysis. 
# This resulted in the creation of four new features: log_carb_fiber_ratio, log_meal_risk_score, log_glycemic_load, and log_insulin_dose.
#--------------------------------------------------------------------------------------------------------------------------------------------------

# Other new features 
# Carb_protein_ratio since protein slows gastric emptying and can moderate glucose absorption
df['carb_protein_ratio'] = df['carb_intake'] / (df['protein_intake'] + 1)

# Carb_fat_ratio since fat slows digestion and glucose absorption
df['carb_fat_ratio'] = df['carb_intake'] / (df['fat_intake'] + 1)

# Insulin_coverage_ratio since insulin dose relative to carbonhydrate intake is crucial to managing diabetes
df['insulin_coverage'] = (df['insulin_dose']
                          /(df['carb_intake']+1)
)
# Activity-adjusted glycemic load. since physical activity improves glucose uptake
df['activity_adjusted_gl'] = (
    df['glycemic_load']
    / (df['physical_activity'] + 1)
)

# Metabolic Risk Score. This combines obesity, stress and recovery
df['metabolic_risk'] = (
    df['bmi']
    * df['stress_level']
    / df['sleep_hours']
)

# Confirm engineered features in dataframe
print(df.head())
#--------------------------------------------------------------------------------------------------------------------------------------------------
# To provide the model with additional information beyond the original variables, several new features were created. 
# The carb_protein_ratio and carb_fat_ratio features were designed to capture the balance between carbohydrates and nutrients 
# that can slow down the rise in blood glucose after a meal. An insulin_coverage feature was created to assess 
# whether the amount of insulin taken was appropriate for the amount of carbohydrates consumed. 
# The activity_adjusted_gl feature combines glycemic load and physical activity to reflect the potential impact of exercise on glucose regulation. 
# Finally, a metabolic_risk feature was developed by combining body weight status (BMI), stress levels, and 
# sleep duration to provide an overall indication of factors that may influence an individual's glucose response. 
# These engineered features were created to help the model better understand the complex interactions that contribute to glucose spikes.
#--------------------------------------------------------------------------------------------------------------------------------------------------

# %%
# FEATURE SELECTION (Based on Domain Knowledge)
#--------------------------------------------------------------------------------------------------------------------------------------------------
# Review Multicollinearity analysis above and compare with the target, those features that were highly correlated

# Update the numeric cols with the continuous engineered cols
numeric_cols.extend([
    'log_carb_fiber_ratio',
    'log_meal_risk_score',
    'log_glycemic_load',
    'log_insulin_dose',
    'carb_protein_ratio',
    'carb_fat_ratio',
    'insulin_coverage',
    'activity_adjusted_gl',
    'metabolic_risk'
])

print(numeric_cols)

# Get correlation
print((df[numeric_cols + ['glucose_spike']].corr()['glucose_spike']).sort_values(ascending=False))

# correlate the log trasform of carb_fibre ratio with meal risk score since it shows it has higher correlation with target than meal risk score
print(df[['log_carb_fiber_ratio', 'meal_risk_score']].corr())

# Drop carb_fiber_ratio  as compared to its highly correlated pair (meal risk score), it has the less correlation with target
df = df.drop('carb_fiber_ratio', axis=1)

# Address pottential data leakage
df = df.drop(['post_meal_glucose', 'glucose_change'], axis=1)

# Confirm Drop
print(df.head())


# %%
# DATA PREPROCESSING
#-------------------------------------------------------------------------------------------------------------------------------------------------
# Binary Mapping
df['gender'] = df['gender'].map({'Female':0, 'Male':1})
df['smoking_status'] = df['smoking_status'].map({'No':0, 'Yes':1})
df['alcohol_consumption'] = df['alcohol_consumption'].map({'No':0, 'Yes':1})
df['diabetes_type'] = df['diabetes_type'].map({'Type 1':0, 'Type 2':1})

# Confirm mapping
print(df.head())

df.info()

# Data Splitting
# separate the target from the features
X = df.drop('glucose_spike', axis=1)
y = df['glucose_spike']

# Confirm separation
print(X)
print(y)

# Split dataset into train and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
 
# Confirm split
print(X_train.head())
print(X_test.head())
print(y_train.head())
print(y_test.head())

# Preprocessing Pipeline for encoding, and selective scaling of features for different models
# Create list of categorical column(s). these will need to be encoded in the pipeline
categoric_cols = ['meal_time']
# View list
print(categoric_cols)

# Confirm list of numeric_cols created previously. These will need to be scaled in the pipeline
print(numeric_cols)

# Since we have dropped carb fiber ratio, post meal glucose and glucose change, we need to creat a differend list of this update 
numeric_cols_updated = [
    col for col in numeric_cols
    if col not in ['carb_fiber_ratio', 'post_meal_glucose', 'glucose_change']
]

# View list
print(numeric_cols_updated)

# Create two preprocessor pipeline of column transformers. This act like a column router, routing columns to appropriate transformer for trasformation

# First preprocessor (transformer) includes the scaling stage
preprocessor_with_scaling = ColumnTransformer(
    transformers=[
        ('encoder', OneHotEncoder(drop='first', handle_unknown='ignore'), categoric_cols),
        ('scaler', StandardScaler(), numeric_cols_updated)
    ],
    remainder='passthrough'
)

# Second preprocessor (transformer) doesn't include the scaling stage
preprocessor_without_scaling = ColumnTransformer(
    transformers=[
        ('encoder', OneHotEncoder(drop='first', handle_unknown='ignore'), categoric_cols)
    ],
    remainder='passthrough'
)

# %%
# MODEL BUILDING : Build Model Pipelines
#-----------------------------------------------------------------------------------------------------------------------------------------------------
# Logistic Regression Pipeline
lr_pipeline = Pipeline([
    ('preprocessor', preprocessor_with_scaling),
    ('model', LogisticRegression(random_state=42, max_iter=5000))
])

# Random Forest Classifier Pipeline
rf_pipeline = Pipeline([
    ('preprocessor', preprocessor_without_scaling),
    ('model', RandomForestClassifier(random_state=42))
])

# XGBoost Classifier Pipeline
xgb_pipeline = Pipeline([
    ('preprocessor', preprocessor_without_scaling),
    ('model', XGBClassifier(random_state=42))
])

# Support Vector Classifier Pipeline
svc_pipeline = Pipeline([
    ('preprocessor', preprocessor_with_scaling),
    ('model', SVC(probability=True, random_state=42))
])

# K-Nearest Neighbors Classifier Pipeline
knn_pipeline = Pipeline([
    ('preprocessor', preprocessor_with_scaling),
    ('model', KNeighborsClassifier())
])

# Create a dictionary to hold the models (model name and model) to be trained
models = {
    'Logistic Regression': lr_pipeline,
    'Random Forest': rf_pipeline,
    'XGBoost': xgb_pipeline,
    'SVC': svc_pipeline,
    'KNN': knn_pipeline
}
# %%
# MODEL TRAINING: Using 5-fold cross-validation for each model
#----------------------------------------------------------------------------------------------------------------------------------------------------
#  Using the cross_validate as we will be comparing models performance accross multiple metrics

# Initialise an empty dictionary for saving the training results for the models
results = {}
# Create a list for the summary of these result
summary =[]
for model_name, model in models.items():
    scores = cross_validate(
        model,
        X_train,
        y_train,
        cv=5,
        scoring=['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    )
    results[model_name] = scores

# Store summarised results (mean and standard deviation) for each model in a dataframe
for model_name, scores in results.items():
    # Create a row for each model
    row = {'model': model_name}
    for metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']:
        row[f"{metric}_mean"] = scores[f"test_{metric}"].mean()
        row[f"{metric}_std"] = scores[f"test_{metric}"].std()
    
    summary.append(row)
    
summary_df = pd.DataFrame(summary)
print(summary_df)

#----------------------------------------------------------------------------------------------------------------------------------------------------
# I chose five classification algorithms—Logistic Regression, K-Nearest Neighbours (KNN), 
# Support Vector Machine (SVM), Random Forest, and XGBoost—to determine 
# which approach was most effective at predicting glucose spike events. 
# These models were selected because they learn from data in different ways, 
# ranging from simple and interpretable techniques to more advanced ensemble methods. 
# By comparing a diverse set of algorithms, 
# I was able to assess how different modelling approaches performed on the dataset and 
# identify the model best suited to capturing the factors associated with glucose spikes.
#----------------------------------------------------------------------------------------------------------------------------------------------------
# KEY OBSERVATION FROM MODEL TRAINING
#----------------------------------------------------------------------------------------------------------------------------------------------------
# On evaluating the five selected classifiers using 5-fold cross-validation, including Logistic Regression, 
# Random Forest, XGBoost, Support Vector Machine (SVM), and K-Nearest Neighbours (KNN). 
# Logistic Regression achieved the strongest overall performance, recording the highest Accuracy (76.3%), 
# Precision (74.1%), Recall (75.1%), F1-score (74.6%), and ROC-AUC (84.8%). Random Forest and 
# SVM produced competitive results but did not surpass Logistic Regression, 
# while XGBoost and KNN demonstrated comparatively lower predictive performance. 
# Based on these results, Logistic Regression was selected as the leading candidate model for further evaluation and tuning, 
# with Random Forest retained as a strong benchmark model.
#----------------------------------------------------------------------------------------------------------------------------------------------------
# %%
# FEATURE IMPORTANCE ANALYSIS
#----------------------------------------------------------------------------------------------------------------------------------------------------
best_model_v1 = lr_pipeline

# Fit the best model on the train set
best_model_v1.fit(X_train, y_train)

# Run permutation importance 
feat_importance = permutation_importance(
    best_model_v1,
    X_train,
    y_train,
    n_repeats=10,
    random_state=42
)

# Create a dataframe of the feature importance
importance = pd.DataFrame({
    'feature':X_train.columns,
    'importance':feat_importance.importances_mean
})

# Sort the features in order of importance
importance = importance.sort_values(
    by='importance',
    ascending=False
)

print(importance)

# ----------------------------------------------------------------------------------------------------------------------------------------------------
# FEATURE IMPORTANCE INSIGHT

# Permutation importance analysis identified carbohydrate-related features as the strongest
# predictors of glucose spike occurrence. Carb_intake emerged as the most influential feature,
# followed by log_carb_fiber_ratio and log_glycemic_load. Lifestyle and physiological factors
# such as fiber_intake, physical_activity, stress_level, and insulin-related measures also
# contributed meaningfully to model performance.
#
# The results further showed that the logarithmically transformed features generally provided
# greater predictive value than their original counterparts, confirming the effectiveness of
# the feature engineering process in reducing skewness and improving signal detection.
#
# Features with very low importance scores, including activity_adjusted_gl, carb_protein_ratio,
# alcohol_consumption, medication_adherence, meal_time, sleep_hours, pre_meal_glucose,
# water_intake, carb_fat_ratio, and metabolic_risk, were identified as candidates for removal.

# Also, the log transform of carb_fiber_ratio, glycemic_load and insulin dose are seen to have a higher importance
# score than their original while the original meal risk score feature score has higher importanc score
# than its log transform, hence, the less performing feature of these pairs will be dropped. This is so to prevent multicollinearity

# %%
# FEATURE SELECTION BASED ON PERMUTATION IMPORTANCE AND DOMAIN KNOWLEDGE
#-----------------------------------------------------------------------------------------------------------------------------------------------------
# Drop some features having permutation importance score less than 0.002 and that are less clinically meaningful
features_to_drop = [
    'log_meal_risk_score',
    'glycemic_load',
    'insulin_dose',
    'activity_adjusted_gl',
    'carb_protein_ratio',
    'alcohol_consumption',
    'medication_adherence',
    'meal_time',
    'sleep_hours',
    'water_intake',
    'carb_fat_ratio',
    'metabolic_risk'
]

# Drop these features X_train and X_test
X_test = X_test.drop(columns=features_to_drop)
X_train = X_train.drop(columns=features_to_drop)

# Confirm drop
print(X_test.shape)
print(X_train.shape)


# Update the numeric column and categorical column list for the preprocessor
numeric_cols_updated = [
    col for col in numeric_cols_updated 
    if col not in features_to_drop
]
# Since Logistic Regression is our best model and it is sensitive to scaling, 
# I will add the initially binary mapped features to the numerical features so they can be scaled together with the other numerical columns

numeric_cols_updated.extend([
    'gender',
    'smoking_status',
    'diabetes_type'
])

categoric_cols = [
    col for col in categoric_cols
    if col not in features_to_drop
]

# Confirm update
print(numeric_cols_updated)
print(categoric_cols)

#-------------------------------------------------------------------------------------------------------------------------------------------------------
# FEATURE SELECTION INSIGHT
# To simplify the model while retaining predictive performance, 
# Features with permutation importance scores below 0.002 were reviewed for potential removal.
# However, feature selection was based on both predictive contribution and domain relevance.
# Variables such as pre_meal_glucose, bmi, glycemic_load, gender, smoking_status, and
# portion_size were retained despite their low importance scores because they represent
# clinically meaningful, behavioural, or demographic factors known to influence glucose
# regulation and diabetes outcomes.
#
# In contrast, features with very low importance and limited clinical or business relevance
# were identified as safe candidates for removal, as they contributed little additional
# predictive value to the model.
# ------------------------------------------------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------------------------------------------------
# Following the features selection where the only categorical feature (meal_time) was dropped
# all remaining features are numerical. Consequently, no further categorical encoding is thus required, 
# hence the processors will be adjusted appropriately
#-------------------------------------------------------------------------------------------------------------------------------------------------------

#%%
# REBUILD ALL PIPELINES
#-------------------------------------------------------------------------------------------------------------------------------------------------------
# Rebuild preprocessor pipelines with updated numerical and categorical columns

preprocessor_with_scaling = ColumnTransformer(
    transformers=[
        ('scaler', StandardScaler(), numeric_cols_updated)
    ],
    remainder='passthrough'
)

preprocessor_without_scaling = ColumnTransformer(
    transformers=[],
    remainder='passthrough'
)

# Rebuild the required model pipeline
lr_pipeline = Pipeline([
    ('preprocessor', preprocessor_with_scaling),
    ('model', LogisticRegression(random_state=42, max_iter=5000))
])

rf_pipeline = Pipeline([
    ('preprocessor', preprocessor_without_scaling),
    ('model', RandomForestClassifier(random_state=42))
])

# Assign these to variable names
best_model_v2 = lr_pipeline
next_best_model = rf_pipeline

# Rebuild the models dictionary
models = {
    'Logistic Regression': best_model_v2,
    'Random Forest': next_best_model
}

# %%
# MODEL RE-TRAINING
#--------------------------------------------------------------------------------------------------------------------------------------------------------
# Rerun cross validation for these two models

# Initiate an empty result dictionary for saving training result
results2 = {}

# Create List for storing summary of result
summary2 = []

for model_name, model in models.items():
    scores = cross_validate(
        model,
        X_train,
        y_train,
        cv=5,
        scoring=['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    )
    results2[model_name]=scores

# Update summary list with results
for model_name, scores in results2.items():
    row = {model:model_name}
    for metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']:
        row[f"{metric}_mean"] = scores[f"test_{metric}"].mean()
        row[f"{metric}_std"] = scores[f"test_{metric}"].std()

    summary2.append(row)

# Create a dataframe of the summary result
summary2_df = pd.DataFrame(summary2)
print(summary2_df)
#---------------------------------------------------------------------------------------------------------------------------------------------------------
# MODEL SELECTION INSIGHT
# Following feature selection, Logistic Regression demonstrated improved performance across
# all evaluation metrics, while Random Forest experienced a slight decline. The results
# indicate that the reduced feature set is better suited to Logistic Regression and that
# the removed features were largely contributing noise rather than useful predictive
# information. Given its superior Accuracy, Precision, Recall, F1-score, and ROC-AUC,
# Logistic Regression was selected as the final candidate model for further refinement.
#---------------------------------------------------------------------------------------------------------------------------------------------------------

# %%
# HYPERPARAMETER TUNING
#---------------------------------------------------------------------------------------------------------------------------------------------------------

# define parameter to tune
param_grid = {
    'model__C': [0.001, 0.01, 0.1, 1, 10, 100],
    'model__penalty': ['l1', 'l2'],
    'model__solver': ['liblinear']
}
# search for best combination of parameters that gives best model performance
grid_search = GridSearchCV(
    estimator=best_model_v2,
    param_grid=param_grid,
    cv=5,
    scoring='f1',
    n_jobs=-1
)

# Fit grid search to the training set
grid_search.fit(X_train, y_train)

# Voe best parameters 
print(grid_search.best_params_)
print(grid_search.best_score_)

# Save best parameters to a dictionary
best_params = grid_search.best_params_
best_score = grid_search.best_score_

# Save to a text file
with open('best_logistic_parameter.txt', 'w') as f:
    f.write(f"Best Parameter:\n{grid_search.best_params_}\n\n")
    f.write(f"Best CV F1 Score:\n{grid_search.best_score_}")

# Create tuned model
best_model_v3 = grid_search.best_estimator_

# Save final verson of trained model
joblib.dump(
    best_model_v3,
    'best_logistic_regression_model.pki'
)
#-----------------------------------------------------------------------------------------------------------------------------------------------------------
# Subsequent hyperparameter tuning further improved model performance,
# indicating that the selected feature set was sufficiently informative for prediction.
# Additional feature removal was not pursued in order to preserve clinically meaningful
# predictors and reduce the risk of over-optimising the model to the training data.
# The tuned Logistic Regression model was therefore retained for final evaluation on the
# unseen test set.
#-----------------------------------------------------------------------------------------------------------------------------------------------------------

# %%
# MODEL TESTING AND EVALUATION
#-----------------------------------------------------------------------------------------------------------------------------------------------------------
# Fit final trained model on the full train set
best_model_v3.fit(X_train, y_train)

# Make prediction on the unseen test set
y_pred = best_model_v3.predict(X_test)

# For ROC-AUC, probabilities are also needed
y_pred_prob = best_model_v3.predict_proba(X_test)[:, 1]


# Calculation of the evaluation metrics
print('Accuracy:', accuracy_score(y_test, y_pred))
print('Precision:', precision_score(y_test, y_pred))
print('Recall:', recall_score(y_test, y_pred))
print('F1 Score:', f1_score(y_test, y_pred))
print('ROC-AUC:', roc_auc_score(y_test, y_pred_prob))

#-----------------------------------------------------------------------------------------------------------------------------------------------------------    
# FINAL MODEL EVALUATION
# The final tuned Logistic Regression model achieved an Accuracy of 76.0%, Precision of 72.1%, Recall of 78.6%, 
# F1-score of 75.2%, and ROC-AUC of 0.848 on the unseen test set. The strong Recall indicates that 
# the model successfully identified the majority of glucose spike events, 
# while the F1-score demonstrates a balanced trade-off between Precision and Recall. Furthermore, 
# the similarity between the cross-validation and test set results suggests that 
# the model generalised well to unseen data and was not significantly affected by overfitting. Overall, 
# the model demonstrated good discriminatory power and provides a reliable approach for predicting glucose spike occurrence.
#-----------------------------------------------------------------------------------------------------------------------------------------------------------

# %%
# RESULT VISUALISATION
#-----------------------------------------------------------------------------------------------------------------------------------------------------------
# Generate and visualise the confusion martrix

cm = confusion_matrix(y_test, y_pred)
print(cm)

plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix of Predicted vs Actual')
plt.show()

#-----------------------------------------------------------------------------------------------------------------------------------------------------------
# cONFUSION MATRIX OBSERVATION

# The confusion matrix demonstrated that the final Logistic Regression model correctly classified 396 non-spike events and 
# 364 glucose spike events. The model produced 141 false positives and 99 false negatives, 
# indicating that it was more likely to generate an unnecessary spike alert than to miss an actual spike. 
# This behaviour aligns with the objective of glucose spike prediction, 
# where detecting genuine spike events is generally more important than avoiding all false alarms. 
# The relatively low number of missed spikes contributed to the model's strong Recall of 78.6%, 
# demonstrating its effectiveness in identifying high-risk glucose responses.
#-----------------------------------------------------------------------------------------------------------------------------------------------------------

# Generate the classification report
print(classification_report(y_test, y_pred))

# Plot the ROC Curve
fpr, tpr, thresholds = roc_curve(
    y_test,
    y_pred_prob
)

plt.figure(figsize=(6,4))
plt.plot(fpr, tpr)
plt.plot([0,1], [0,1], '--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()
# Generate a detailed classification report for best model selected

#-----------------------------------------------------------------------------------------------------------------------------------------------------------
# CLASSIFICATION REPORT INSIGHT
# The classification report indicates that the model achieved balanced performance across both classes, 
# with F1-scores of 0.77 for non-spike events and 0.75 for glucose spike events. 
# The higher Recall (79%) for the glucose spike class demonstrates the model's strong ability to identify actual spike occurrences, 
# while maintaining reasonable Precision (72%), making it suitable for detecting high-risk glucose responses.

# ROC CURVE INSIGHT
# The ROC curve remained substantially above the random-classification baseline, 
# confirming the model's strong ability to distinguish between glucose spike and non-spike events across different decision thresholds. 
# This is supported by the ROC-AUC score of 0.848, indicating good discriminatory performance and robust classification capability.
#-----------------------------------------------------------------------------------------------------------------------------------------------------------

#-----------------------------------------------------------------------------------------------------------------------------------------------------------
# KEY RISK FACTORS AND RECOMMENDED ACTIONS
#-----------------------------------------------------------------------------------------------------------------------------------------------------------
# Monitor carbohydrate intake closely
 # Carbohydrate intake was the most influential predictor of glucose spikes. 
 # Individuals should pay close attention to the amount of carbohydrates consumed at each meal.

# Increase fibre intake when consuming carbohydrates
 # The carbohydrate-to-fibre ratio was one of the strongest risk factors, 
 # suggesting that pairing carbohydrates with fibre-rich foods may help reduce spike risk.

# Choose meals with a lower glycaemic load
 # Since glycaemic load was a major predictor, 
 # selecting foods that release glucose more gradually may help improve blood sugar control.

# Encourage regular physical activity
 # Physical activity was identified as 
 # an important factor in the model and may help reduce the likelihood of post-meal glucose spikes.

# Promote stress management strategies
 # Stress level ranked among the top predictors, highlighting the potential value of stress reduction techniques 
 # such as exercise, relaxation activities, and adequate rest.
#----------------------------------------------------------------------------------------------------------------------------------------------------------

