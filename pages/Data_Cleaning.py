
import streamlit as st
import pandas as pd
import numpy as np
import os
from glob import glob

st.set_page_config(page_title="Data Cleaning", layout="wide")

st.sidebar.success("This Data Cleaning will show the kind of techniques employed")

st.title('Data Cleaning and Preprocessing')
st.write('Upload your dataset(s) and apply data cleaning steps.')

# --- File Upload Section ---
uploaded_files = st.file_uploader("Upload your CSV files", type=['csv'], accept_multiple_files=True)

@st.cache_data
def load_and_clean_data(files):
    """Reads, merges, and cleans the uploaded CSV files."""
    if not files:
        return None, "Please upload data files to begin cleaning."

    dfs = []
    for file in files:
        df = pd.read_csv(file, parse_dates=['Date'], low_memory=False)
        dfs.append(df)

    dataset = pd.concat(dfs, ignore_index=True)

   
    duplicates = dataset.duplicated().sum()
    if duplicates > 0:
        dataset = dataset.drop_duplicates()

   
    dataset['Date'] = pd.to_datetime(dataset['Date'], errors='coerce')


    median_date = dataset['Date'].median()
    dataset['Date'].fillna(median_date, inplace=True)

    # Handle missing values in pollutant columns using median
    pollutants = ['PM2.5','PM10','NO','NO2','NOx','NH3','CO','SO2','O3','Benzene','Toluene','Xylene']
    for col in pollutants:
        median_val = dataset[col].median()
        dataset[col] = dataset[col].fillna(median_val)

    # Handle missing values in AQI columns using median
    median_val = dataset["AQI"].median()
    dataset["AQI"] = dataset["AQI"].fillna(median_val)

    # Impute missing AQI_Bucket values using AQI ranges
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
    
    return dataset, f"Data cleaned and merged successfully. Final shape: {dataset.shape}"

# --- Display Results ---
if uploaded_files:
    data_frame, status_message = load_and_clean_data(uploaded_files)
    st.success(status_message)
    
    st.subheader("Cleaned Dataset Preview")
    st.dataframe(data_frame.head(10))

    st.subheader("Missing Values After Cleaning")
    st.dataframe(data_frame.isnull().sum().sort_values(ascending=False))

    st.subheader("Data Cleaning Summary")
    st.write("""
    The dataset was cleaned and prepared for analysis through several important preprocessing steps. Duplicate rows were removed to maintain data accuracy, and the Date column was converted into a proper datetime format. Missing dates were filled using the median timestamp. Several key pollutant columns (PM2.5, PM10, etc.) also had missing values, which were handled using median imputation.
    
    A major part of the cleaning involved fixing the **AQI_Bucket** column using a function to re-assign categories based on the numerical AQI ranges, allowing us to keep more data for modeling.
    """)

else:
    st.info("Please upload your data files using the widget above.")

st.markdown('---') 
st.write('Reading files directly from a fixed local folder path.')

# Correct relative path
folder_path = "Assessment Data-20251028"

@st.cache_data
def load_and_clean_data_from_folder(f_path):
    
    if not os.path.exists(f_path):
        return None, f"Error: Folder not found at {f_path}"

    csv_files = glob(os.path.join(f_path, "*.csv"))

    if not csv_files:
        return None, f"Error: No CSV files found in {f_path}"

    dfs = [pd.read_csv(file, parse_dates=['Date'], low_memory=False) for file in csv_files]
    dataset = pd.concat(dfs, ignore_index=True)

    # --- SAME CLEANING LOGIC AS BEFORE ---
    duplicates = dataset.duplicated().sum()
    if duplicates > 0:
        dataset = dataset.drop_duplicates()

    dataset['Date'] = pd.to_datetime(dataset['Date'], errors='coerce')
    median_date = dataset['Date'].median()
    dataset['Date'].fillna(median_date, inplace=True)
    
    pollutants = ['PM2.5','PM10','NO','NO2','NOx','NH3','CO','SO2','O3','Benzene','Toluene','Xylene']
    for col in pollutants:
        median_val = dataset[col].median()
        dataset[col] = dataset[col].fillna(median_val)

    median_val = dataset["AQI"].median()
    dataset["AQI"] = dataset["AQI"].fillna(median_val)

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
    
    return dataset, f"Data cleaned and merged successfully. Final shape: {dataset.shape}"


# --- Display Results ---
data_frame, status_message = load_and_clean_data_from_folder(folder_path)

if data_frame is not None:
    st.success(status_message)
    st.subheader("Combined Dataset Preview")
    st.dataframe(data_frame.head(10))

    st.markdown('---')
    st.subheader("Cleaning Results for Individual Cities")

   
    unique_cities = data_frame['City'].unique()
    cities_to_display = unique_cities[:6] 

  
    for city_name in cities_to_display:
        st.markdown(f"**City:** *{city_name}*")
        city_df = data_frame[data_frame['City'] == city_name]
        st.dataframe(city_df.head(6))
        st.write(f"Total records for {city_name}: {len(city_df)}")
        st.markdown("---") 

else:
    st.error(status_message)


st.markdown('---') 

# Create two columns for the text areas and buttons
col1, col2 = st.columns(2)

with col1:
    st.subheader('Upload Cleaning Notes ')
   
    text_area_1 = st.text_area("Enter notes for Upload Data Cleaning :", height=200, key="ta1")
    
    if st.button('Publish Notes', key="btn1"):
        if text_area_1:
            st.info(f"Step 1 applied with input: {text_area_1[:50]}...")
        else:
            st.warning("Please enter text for Step 1.")

with col2:
    st.subheader('Fixed Cleaning Notes')

    text_area_2 = st.text_area("Enter notes for Fixed Data Cleaning :", height=200, key="ta2")
 
    if st.button('Publish Notes', key="btn2"):
        if text_area_2:
            st.info(f"Step 2 applied with input: {text_area_2[:50]}...")
        else:
            st.warning("Please enter text for Step 2.")



