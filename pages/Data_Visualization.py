import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import os
from glob import glob
import plotly.graph_objects as go

# STREAMLIT PAGE SETTINGS
st.set_page_config(page_title="Data Visualization", layout="wide")
st.title('Data Visualization and Feature Engineering')
st.sidebar.success("This Visualization will show the kind of techniques employed")

# IMPORTANT: USE RELATIVE PATH (WORKS ON STREAMLIT CLOUD)
folder_path = "Assessment Data-20251028"

# -----------------------------------------------------------
# LOAD, CLEAN, AND FEATURE-ENGINEER DATA
# -----------------------------------------------------------
@st.cache_data
def load_clean_and_engineer_data(f_path):

    # Check folder exists
    if not os.path.exists(f_path):
        return None, f" Error: Folder not found at '{f_path}'. Ensure the folder is inside your GitHub repository."

    # Find CSV files
    csv_files = glob(os.path.join(f_path, "*.csv"))
    if not csv_files:
        return None, f" Error: No CSV files found in '{f_path}'."

    # Load CSV files
    dfs = []
    for file in csv_files:
        df = pd.read_csv(file, parse_dates=['Date'], low_memory=False)
        dfs.append(df)

    dataset = pd.concat(dfs, ignore_index=True)

    # --- DATA CLEANING ---
    dataset = dataset.drop_duplicates()
    dataset['Date'] = pd.to_datetime(dataset['Date'], errors='coerce')
    dataset['Date'].fillna(dataset['Date'].median(), inplace=True)

    pollutants = ['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3',
                  'Benzene', 'Toluene', 'Xylene']
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

    dataset['AQI_Bucket'] = dataset.apply(
        lambda row: assign_bucket(row['AQI']) if pd.isna(row['AQI_Bucket']) else row['AQI_Bucket'],
        axis=1
    )

    # --- FEATURE ENGINEERING ---
    dataset['Year'] = dataset['Date'].dt.year
    dataset['Month'] = dataset['Date'].dt.month
    dataset['DayOfWeek'] = dataset['Date'].dt.dayofweek
    dataset['Week'] = dataset['Date'].dt.isocalendar().week.astype(int)

    def get_season(month):
        if month in [12, 1, 2]: return "Winter"
        elif month in [3, 4, 5]: return "Summer"
        elif month in [6, 7, 8]: return "Monsoon"
        else: return "Post-Monsoon"

    dataset['Season'] = dataset['Month'].apply(get_season)

    dataset['Total_Nitrogen'] = dataset['NO'] + dataset['NO2'] + dataset['NOx']
    dataset['Total_VOC'] = dataset['Benzene'] + dataset['Toluene'] + dataset['Xylene']
    dataset['PM_Load'] = dataset['PM2.5'] + dataset['PM10']

    dataset = dataset.sort_values(['City', 'Date'])

    # Rolling averages
    dataset['PM25_7day_avg'] = dataset.groupby('City')['PM2.5'].transform(lambda x: x.rolling(7, min_periods=1).mean())
    dataset['PM10_7day_avg'] = dataset.groupby('City')['PM10'].transform(lambda x: x.rolling(7, min_periods=1).mean())

    # Lag features
    dataset['PM25_lag1'] = dataset.groupby('City')['PM2.5'].shift(1)
    dataset['PM10_lag1'] = dataset.groupby('City')['PM10'].shift(1)

    dataset['PM25_lag1'].fillna(dataset['PM25_lag1'].median(), inplace=True)
    dataset['PM10_lag1'].fillna(dataset['PM10_lag1'].median(), inplace=True)

    # Encoding
    dataset['City_Code'] = dataset['City'].astype('category').cat.codes
    dataset['Season_Code'] = dataset['Season'].astype('category').cat.codes

    return dataset, " Dataset successfully loaded and processed!"


# Load dataset
df_engineered, message = load_clean_and_engineer_data(folder_path)

if df_engineered is None:
    st.error(message)
    st.stop()

st.success(message)

# -----------------------------------------------------------
# DATA PREVIEW
# -----------------------------------------------------------
st.subheader("Engineered Features Preview")
st.dataframe(df_engineered[['Date', 'Season', 'Total_Nitrogen', 'PM25_lag1', 'City_Code']].head(10))

# INFO + DESCRIBE SECTION
st.subheader("Dataset Info & Summary Statistics")
with st.expander("View dataset .info() and .describe() output"):
    from io import StringIO
    buffer = StringIO()
    df_engineered.info(buf=buffer)
    st.text(buffer.getvalue())
    st.write(df_engineered.describe())

# -----------------------------------------------------------
# MISSING VALUES HEATMAP
# -----------------------------------------------------------
st.subheader("Missing Values Heatmap (After Feature Engineering)")
fig, ax = plt.subplots(figsize=(14, 6))
sns.heatmap(df_engineered.isnull(), cbar=False, cmap='viridis', ax=ax)
st.pyplot(fig)

# -----------------------------------------------------------
# TREND PLOT
# -----------------------------------------------------------
st.subheader("Interactive PM2.5 Trend by City")
city_selection = st.selectbox("Select a City:", df_engineered['City'].unique())

city_data = df_engineered[df_engineered['City'] == city_selection]
fig_px = px.line(city_data, x='Date', y='PM2.5', title=f"Daily PM2.5 Trends in {city_selection}")
st.plotly_chart(fig_px, use_container_width=True)

# -----------------------------------------------------------
# AQI CATEGORY BAR CHART
# -----------------------------------------------------------
st.subheader("AQI Categories Distribution")
aqi_counts = df_engineered['AQI_Bucket'].value_counts().reset_index()
aqi_counts.columns = ['AQI_Bucket', 'Count']

fig_aqi = px.bar(aqi_counts, x='AQI_Bucket', y='Count', color='AQI_Bucket',
                 title="Distribution of AQI Categories", template="plotly_white")
st.plotly_chart(fig_aqi, use_container_width=True)

# -----------------------------------------------------------
# PM2.5 vs AQI SCATTER
# -----------------------------------------------------------
st.subheader("PM2.5 vs AQI Relationship")
fig = px.scatter(df_engineered, x="PM2.5", y="AQI", opacity=0.5,
                 title="PM2.5 vs AQI Relationship", template="plotly_white",
                 trendline="ols")
st.plotly_chart(fig, use_container_width=True)

# -----------------------------------------------------------
# POLLUTANT AVERAGES BAR CHART
# -----------------------------------------------------------
st.subheader("Average Concentration of Pollutants")
pollutants = ['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3', 'Benzene', 'Toluene', 'Xylene']
pollutant_means = df_engineered[pollutants].mean().reset_index()
pollutant_means.columns = ["Pollutant", "Average"]

fig = px.bar(pollutant_means, x="Pollutant", y="Average",
             title="Average Concentration of Each Pollutant",
             template="plotly_white", color="Pollutant")
st.plotly_chart(fig, use_container_width=True)

# -----------------------------------------------------------
# CORRELATION HEATMAP
# -----------------------------------------------------------
st.subheader("Correlation Heatmap for Key Pollutants")
corr_matrix = df_engineered[pollutants].corr()

fig = go.Figure(data=go.Heatmap(
    z=corr_matrix.values,
    x=corr_matrix.columns,
    y=corr_matrix.index,
    colorscale="RdBu",
    zmin=-1,
    zmax=1
))
fig.update_layout(title="Correlation Heatmap of Pollutants", template="plotly_white")
st.plotly_chart(fig, use_container_width=True)

# -----------------------------------------------------------
# SUNBURST: AQI BY SEASON & CITY
# -----------------------------------------------------------
st.subheader("AQI Distribution by Season and City")
aqi_sunburst = df_engineered.groupby(["Season", "City"])["AQI"].mean().reset_index()

fig = px.sunburst(aqi_sunburst, path=["Season", "City"], values="AQI",
                  title="AQI Distribution by Season and City")
st.plotly_chart(fig, use_container_width=True)

# -----------------------------------------------------------
# MONTHLY PM2.5 TREND
# -----------------------------------------------------------
st.subheader("Monthly PM2.5 Average Trend")
monthly_avg = df_engineered.groupby('Month')['PM2.5'].mean().reset_index()
fig = px.line(monthly_avg, x='Month', y='PM2.5', markers=True,
              title="Average PM2.5 Levels by Month", template="plotly_white")
st.plotly_chart(fig, use_container_width=True)

# -----------------------------------------------------------
# PM2.5 VARIATION ACROSS CITIES
# -----------------------------------------------------------
st.subheader("PM2.5 Variation Across Cities")
dataset_city = df_engineered.groupby('City')['PM2.5'].mean().reset_index()

fig = px.box(dataset_city, x="City", y="PM2.5",
             title="PM2.5 Levels Across Cities", template="plotly_white")
fig.update_layout(xaxis={'categoryorder': 'total descending'})
st.plotly_chart(fig, use_container_width=True)
