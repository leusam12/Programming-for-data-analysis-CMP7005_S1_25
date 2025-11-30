import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import os
from glob import glob
import plotly.graph_objects as go 



st.set_page_config(page_title="Data Visualization", layout="wide")

st.title('Data Visualization and Feature Engineering')

st.sidebar.success("This Visualization will show the kind of techniques employed")


folder_path = "C:\\Users\\Jide\\Desktop\\akestreamlit\\Assessment Data-20251028\\"


@st.cache_data
def load_clean_and_engineer_data(f_path):
 
    
    if not os.path.exists(f_path):
        st.error(f"Error: Folder not found at {f_path}")
        st.stop()
    csv_files = glob(os.path.join(f_path, "*.csv"))
    if not csv_files:
        st.error(f"Error: No CSV files found in {f_path}")
        st.stop()

    dfs = []
    for file in csv_files:
        df = pd.read_csv(file, parse_dates=['Date'], low_memory=False)
        dfs.append(df)
    dataset = pd.concat(dfs, ignore_index=True)

    # --- Data Cleaning Logic ---
    dataset = dataset.drop_duplicates()
    dataset['Date'] = pd.to_datetime(dataset['Date'], errors='coerce')
    median_date = dataset['Date'].median()
    dataset['Date'].fillna(median_date, inplace=True)
    pollutants = ['PM2.5','PM10','NO','NO2','NOx','NH3','CO','SO2','O3','Benzene','Toluene','Xylene']
    for col in pollutants:
        dataset[col] = dataset[col].fillna(dataset[col].median())
    dataset["AQI"] = dataset["AQI"].fillna(dataset["AQI"].median())
    def assign_bucket(aqi):
        if aqi <= 50: return "Good"
        elif aqi <= 100: return "Satisfactory"
        elif aqi <= 200: return "Moderate"
        elif aqi <= 300: return "Poor"
        elif aqi <= 400: return "Very Poor"
        else: return "Severe"
    dataset['AQI_Bucket'] = dataset.apply(lambda row: assign_bucket(row['AQI']) if pd.isna(row['AQI_Bucket']) else row['AQI_Bucket'], axis=1)

    # --- Feature Engineering Logic ---
    dataset['Year'] = dataset['Date'].dt.year
    dataset['Month'] = dataset['Date'].dt.month
    dataset['DayOfWeek'] = dataset['Date'].dt.dayofweek
    dataset['Week'] = dataset['Date'].dt.isocalendar().week.astype(int)
    
    def get_season(month):
        if month in [12, 1, 2]: 
            return "Winter"
        elif month in [3, 4, 5]: 
            return "Summer"
        elif month in [6, 7, 8]:
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
    
    # FIX: Separate definition and fillna for PM25_lag1/PM10_lag1
    dataset['PM25_lag1'] = dataset.groupby('City')['PM2.5'].shift(1)
    dataset['PM10_lag1'] = dataset.groupby('City')['PM10'].shift(1)

    # Now that the columns exist, calculate their medians and fill NAs
    dataset['PM25_lag1'].fillna(dataset['PM25_lag1'].median(), inplace=True)
    dataset['PM10_lag1'].fillna(dataset['PM10_lag1'].median(), inplace=True)

    dataset['City_Code'] = dataset['City'].astype('category').cat.codes
    dataset['Season_Code'] = dataset['Season'].astype('category').cat.codes

    return dataset


df_engineered = load_clean_and_engineer_data(folder_path) 


st.subheader("Engineered Features Preview")
st.dataframe(df_engineered[['Date', 'Season', 'Total_Nitrogen', 'PM25_lag1', 'City_Code']].head(10))
st.subheader("Dataset Info & Summary Statistics")
with st.expander("View dataset .info() and .describe() output"):
    from io import StringIO
    buffer = StringIO()
    df_engineered.info(buf=buffer)
    st.text(buffer.getvalue())
    st.write(df_engineered.describe())
st.subheader("Missing Values Heatmap (After FE)")
fig, ax = plt.subplots(figsize=(14, 6))
sns.heatmap(df_engineered.isnull(), cbar=False, cmap='viridis', ax=ax)
st.pyplot(fig)
st.write("This static heatmap shows where missing values are located across all features after engineering and imputation.")
st.subheader("Interactive Trend Plot (Plotly)")
city_selection = st.selectbox("Select a City to Visualize Trends:", df_engineered['City'].unique())
if city_selection:
    city_data = df_engineered[df_engineered['City'] == city_selection]
    fig_px = px.line(city_data, x='Date', y='PM2.5', title=f"Daily PM2.5 Trends in {city_selection}")
    st.plotly_chart(fig_px, use_container_width=True)


