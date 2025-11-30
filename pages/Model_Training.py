import streamlit as st
import pandas as pd
import numpy as np
import os
from glob import glob
# Machine Learning Imports
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression 




st.set_page_config(page_title="Model Training", layout="wide")

st.title('Model Training and Evaluation')

st.sidebar.success("Welcome to the Machine Learning Page")

# --- Define Local Folder Path (Same as cleaning page) ---
folder_path = "C:\\Users\\Jide\\Desktop\\akestreamlit\\Assessment Data-20251028\\"

# --- Function to load, clean, AND engineer data (copied for independence) ---
@st.cache_data
def load_clean_and_engineer_data(f_path):
    if not os.path.exists(f_path): st.error(f"Error: Folder not found at {f_path}"); st.stop()
    csv_files = glob(os.path.join(f_path, "*.csv"))
    if not csv_files: st.error(f"Error: No CSV files found in {f_path}"); st.stop()
    dfs = []
    for file in csv_files: dfs.append(pd.read_csv(file, parse_dates=['Date'], low_memory=False))
    dataset = pd.concat(dfs, ignore_index=True)
    dataset = dataset.drop_duplicates()
    dataset['Date'] = pd.to_datetime(dataset['Date'], errors='coerce')
    median_date = dataset['Date'].median()
    dataset['Date'].fillna(median_date, inplace=True)
    pollutants = ['PM2.5','PM10','NO','NO2','NOx','NH3','CO','SO2','O3','Benzene','Toluene','Xylene']
    for col in pollutants: dataset[col] = dataset[col].fillna(dataset[col].median())
    dataset["AQI"] = dataset["AQI"].fillna(dataset["AQI"].median())
    def assign_bucket(aqi):
        if aqi <= 50: return "Good"
        elif aqi <= 100: return "Satisfactory"
        elif aqi <= 200: return "Moderate"
        elif aqi <= 300: return "Poor"
        elif aqi <= 400: return "Very Poor"
        else: return "Severe"
    dataset['AQI_Bucket'] = dataset.apply(lambda row: assign_bucket(row['AQI']) if pd.isna(row['AQI_Bucket']) else row['AQI_Bucket'], axis=1)
    
    dataset['Year'] = dataset['Date'].dt.year
    dataset['Month'] = dataset['Date'].dt.month
    dataset['DayOfWeek'] = dataset['Date'].dt.dayofweek
    dataset['Week'] = dataset['Date'].dt.isocalendar().week.astype(int)
    
    # FIX: Corrected indentation and added month lists
    def get_season(month):
        if month in [12, 1, 2]:
            return "Winter"
        elif month in [3, 4, 5, 6]:
            return "Summer"
        elif month in [7, 8]:
            return "Monsoon"
        else:
            return "Post-Monsoon"
            
    dataset['Season'] = dataset['Month'].apply(get_season)   
    dataset['Total_Nitrogen'] = dataset['NO'] + dataset['NO2'] + dataset['NOx']
    dataset['Total_VOC'] = dataset['Benzene'] + dataset['Toluene'] + dataset['Xylene']
    dataset['PM_Load'] = dataset['PM2.5'] + dataset['PM10']
    dataset = dataset.sort_values(['City', 'Date'])
    dataset['PM25_7day_avg'] = dataset.groupby('City')['PM2.5'].transform(lambda x: x.rolling(window=7, min_periods=1).mean())
    dataset['PM10_7day_avg'] = dataset.groupby('City')['PM10'].transform(lambda x: x.rolling(window=7, min_periods=1).mean())
    dataset['PM25_lag1'] = dataset.groupby('City')['PM2.5'].shift(1)
    dataset['PM10_lag1'] = dataset.groupby('City')['PM10'].shift(1)
    dataset['PM25_lag1'].fillna(dataset['PM25_lag1'].median(), inplace=True)
    dataset['PM10_lag1'].fillna(dataset['PM10_lag1'].median(), inplace=True)
    dataset['City_Code'] = dataset['City'].astype('category').cat.codes
    dataset['Season_Code'] = dataset['Season'].astype('category').cat.codes
    return dataset

# Load data independently
df = load_clean_and_engineer_data(folder_path) 

st.subheader("Data Preparation for Modeling")

# --- ML Data Prep from Colab ---

# Define features (X) and target (y)
X = df.drop(columns=['AQI_Bucket', 'Date'])
y = df['AQI_Bucket']

st.write(f"Features shape: {X.shape}, Target shape: {y.shape}")
st.dataframe(X.head())

# Separate categorical and numerical columns for preprocessing
categorical_cols = X.select_dtypes(include=['object', 'category']).columns
numerical_cols  = X.select_dtypes(exclude=['object', 'category']).columns

st.write(f"Numerical columns: {list(numerical_cols)}")
st.write(f"Categorical columns: {list(categorical_cols)}")

st.markdown('---')

st.subheader("Building the Preprocessing Pipeline")

st.markdown("""
We use scikit-learn's `ColumnTransformer` and `Pipeline` to automate preprocessing:
*   **Numerical features** are standardized (scaled).
*   **Categorical features** are One-Hot Encoded.
""")

numerical_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

st.success("Preprocessing pipeline defined successfully.")

