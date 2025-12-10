<<<<<<< HEAD
# Air Quality Analysis and AQI_Bucket Prediction Using Machine Learning

This project focuses on analyzing air quality data collected from multiple Indian cities between 2015 and 2020. The dataset contains daily observations of critical air pollutants such as PM2.5, PM10, NO, NO2, NOx, CO, SO2, NH3, O3, and VOCs (Benzene, Toluene, Xylene), along with the Air Quality Index (AQI) and the categorized AQI_Bucket.

The goal of this project is to clean and analyze the dataset, create new useful features, build machine learning models to predict AQI_Bucket, and design a multi-page Streamlit GUI application for interactive data exploration and predictions.

# Project Objectives

Clean and preprocess the raw dataset.

Perform Exploratory Data Analysis (EDA).

Engineer new features to improve model performance.

Build predictive machine learning models for AQI_Bucket.

Visualize pollutant patterns and relationships.

Develop a multi-page GUI using Streamlit.

Use GitHub for version control and documentation.

# Requirements

import pandas as pd

import os

from glob import glob

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.express as px

import plotly.graph_objects as go

import streamlit

scikit-learn

from google.colab import drive

# Project Objectives
Clean and preprocess the raw dataset.

Perform Exploratory Data Analysis (EDA).

Engineer new features to improve model performance.

Build predictive machine learning models for AQI_Bucket.

Visualize pollutant patterns and relationships.

Develop a multi-page GUI using Streamlit.

Use GitHub for version control and documentation.

# Data Cleaning & Preprocessing
The initial dataset contained:

Over 29,000 rows

16 pollutant and index columns

Missing values in multiple features

Several CSV files requiring merging

# Data Cleaning Steps:
Merged all CSV files from the dataset folder.

Converted the Date column to datetime format.

Removed duplicate rows.

Handled missing values using:

Median imputation for pollutants

Median timestamp for missing dates

Filled missing AQI_Bucket using an AQI-based classification function.

Removed invalid or inconsistent values.

<img width="1920" height="1080" alt="Image" src="https://github.com/user-attachments/assets/9720ae78-3143-40a4-86aa-51a9dab63f41" />

<img width="1920" height="1080" alt="Image" src="https://github.com/user-attachments/assets/272b9963-e43b-407f-a55e-bf14ad1f2b5d" />

# Feature Engineering
To enrich the dataset and improve model performance, the following new features were created:

# Time-based Features

Year

Month

DayOfWeek

Season

# Pollutant Aggregation Features

Total_Nitrogen = NO + NO2 + NOx

Total_VOC = Benzene + Toluene + Xylene

PM_Load = PM2.5 + PM10

# Rolling Window Features

PM2.5_7day_avg

PM10_7day_avg

# Lag Features

PM2.5_lag1

PM10_lag1

These engineered features helped the model detect trends, weekly patterns, and pollutant interactions.

<img width="1920" height="1080" alt="Image" src="https://github.com/user-attachments/assets/85e59036-5da6-4712-b385-cd13c450dc74" />

<img width="1920" height="1080" alt="Image" src="https://github.com/user-attachments/assets/4c5753e6-ab32-49c7-92cc-6f17a17e2b8d" />

<img width="1920" height="1080" alt="Image" src="https://github.com/user-attachments/assets/871062e0-8c21-496c-a9e8-aee0585b8e78" />

# Exploratory Data Analysis (EDA)

Multiple visualizations were generated to understand patterns in the dataset. These include:

# Univariate Analysis

Histograms of pollutants

AQI_Bucket distribution

# Bivariate Analysis

PM2.5 vs AQI scatter plot

VOC pollutant relationships

# Multivariate Analysis

Correlation heatmap

Pairplot of key pollutants

City-wise PM2.5 comparison (Box plot)

Monthly pollutant seasonality trends

Weekday pollutant variation

VOC comparison across cities

These insights helped show which pollutants were most impactful and how air quality changes across time and location.

<img width="1920" height="1080" alt="Image" src="https://github.com/user-attachments/assets/bbfc93fd-775b-4da3-82a2-878241bce77e" />

<img width="1920" height="1080" alt="Image" src="https://github.com/user-attachments/assets/799886f3-2d6a-47a9-aa30-af2b84dbc436" />

<img width="1920" height="1080" alt="Image" src="https://github.com/user-attachments/assets/35733eb1-25d4-4726-b750-a2b0d0165cfa" />

<img width="1920" height="1080" alt="Image" src="https://github.com/user-attachments/assets/225d6e6c-2f62-485d-9122-008ee3245ca7" />

<img width="1920" height="1080" alt="Image" src="https://github.com/user-attachments/assets/412d9272-53bc-4193-a2f5-c68500ecc870" />

<img width="1920" height="1080" alt="Image" src="https://github.com/user-attachments/assets/2931f381-8740-4e7f-aa21-8df38b57ca88" />

<img width="1920" height="1080" alt="Image" src="https://github.com/user-attachments/assets/a73a0573-dade-4cb2-94b2-c0a8b192fc17" />

# Machine Learning Model

The target variable is AQI_Bucket, a categorical classification problem.

# Machine Learning Libraries Import

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score

from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression

# Models Built:

Random Forest Classifier

Logistic Regression (baseline)

# Preprocessing Pipeline:

One-hot encoding of categorical features

Standard scaling for numerical features

Train-test split

Cross-validation

Grid Search hyperparameter tuning

# Random Forest Performance:

Accuracy: ~99%

Strong precision and recall across all AQI categories

Feature importance shows AQI, PM2.5, PM_Load, PM lags, and nitrogen pollutants as top predictors.

# Logistic Regression:

Achieved ~96% accuracy

Served as a good baseline model

Random Forest significantly outperformed Logistic Regression

<img width="1920" height="1080" alt="Image" src="https://github.com/user-attachments/assets/85e78cb9-24f4-490c-a475-9bd4ad4738eb" />

<img width="1920" height="1080" alt="Image" src="https://github.com/user-attachments/assets/5606a0a0-5655-4bcf-8bf8-be6ca605d55f" />

<img width="1920" height="1080" alt="Image" src="https://github.com/user-attachments/assets/ca42dac2-ce7a-4fa8-91ef-82d7c021a6dd" />

<img width="1920" height="1080" alt="Image" src="https://github.com/user-attachments/assets/e98cc166-596c-4990-8cf8-78d17f0ad5b9" />

<img width="1920" height="1080" alt="Image" src="https://github.com/user-attachments/assets/03228e87-b806-45a0-b008-c6d25039a16c" />

<img width="1920" height="1080" alt="Image" src="https://github.com/user-attachments/assets/0dea8d63-313f-4e63-a864-ffda474d64b4" />

<img width="1920" height="1080" alt="Image" src="https://github.com/user-attachments/assets/e98464db-fe54-487e-bd7c-00206c0f47c3" />

<img width="1920" height="1080" alt="Image" src="https://github.com/user-attachments/assets/0b504f29-aa61-4150-9f85-d243fc95039d" />

# Streamlit Multi-Page GUI Application
A professional, user-friendly dashboard was built using Streamlit with 4 pages:

# Page 1 — Data Overview

Shows merged dataset preview

Summary statistics

Data types

Missing value report

General project description

# Page 2 — Data Cleaning

Displays cleaning steps

Before/after comparison

Handling of missing values

Feature engineering summary

# Page 3 — Data Visualization

Interactive plots using Plotly

Monthly and weekly trends

City comparison

Correlation heatmaps

PM distributions

# Page 4 — Machine Learning Model

Shows model accuracy

Classification report

Confusion matrix

Feature importance plot

Prediction interface for user inputs

<img width="1920" height="1080" alt="Image" src="https://github.com/user-attachments/assets/e6209449-e5e9-46cd-b62a-9cc4d654dc4a" />

<img width="1920" height="1080" alt="Image" src="https://github.com/user-attachments/assets/50847304-b2c3-4fb6-a9af-ead5dbb775e9" />

<img width="1920" height="1080" alt="Image" src="https://github.com/user-attachments/assets/0e33e8ee-8419-41c2-833c-595a7bc5f08a" />

<img width="1920" height="1080" alt="Image" src="https://github.com/user-attachments/assets/83d3f7cd-2597-40d0-a661-de9d7d541603" />

<img width="1920" height="1080" alt="Image" src="https://github.com/user-attachments/assets/1bab6863-21b0-4861-a9b3-37c65cb87711" />

# Project Structure
Air-Quality-Project/
│
├── pages/
│   ├── 1_Data_Overview.py
│   ├── 2_Data_Cleaning.py
│   ├── 3_Data_Visualization.py
│   ├── 4_Machine_Learning_Model.py
│
├── data/
│   └── Assessment Data-20251028/
│
├── model/
│   └── rf_model.pkl
│
├── README.md
└── app.py

# How to Run the Project
## Install dependencies:
pip install -r requirements.txt

# Run the Streamlit app:
streamlit run app.py

# Conclusion
This project successfully analyzes air quality patterns, engineers meaningful features, and builds a high-performing machine learning model to predict AQI_Bucket. The Streamlit GUI provides a clean and interactive way to explore, visualize, and predict air quality levels, making the system practical for real-world environmental monitoring or educational purposes.
=======
# Programming-for-data-analysis-CMP7005_S1_25
CMP7005_S1_25
>>>>>>> 4bac1a95d784b24aebeb47081cc2556b3ef421dc
