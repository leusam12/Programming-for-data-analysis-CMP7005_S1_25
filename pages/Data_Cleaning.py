import streamlit as st
import pandas as pd
import numpy as np
import os
from glob import glob
import plotly.express as px

st.set_page_config(page_title="Data Cleaning", layout="wide")

# ---------------- HIDE SIDEBAR ----------------
hide_sidebar = """
<style>
    [data-testid="stSidebar"] {display: none;}
    [data-testid="collapsedControl"] {display: none;}
</style>
"""
st.markdown(hide_sidebar, unsafe_allow_html=True)

# ---------------- TOP NAVIGATION BAR ----------------
navbar = """
<style>
.topnav {
    background-color: #1f2937;
    overflow: hidden;
    padding: 18px;
    border-radius: 10px;
    display: flex;
    justify-content: space-between;
}
.topnav a {
    color: #f2f2f2;
    padding: 14px 26px;
    text-decoration: none;
    font-size: 17px;
    border-radius: 6px;
    transition: 0.3s;
}
.topnav a:hover {
    background-color: #374151;
}
.topnav a.active {
    background-color: #2563eb;
    color: white;
}
.card {
    background: #e962f5;
    padding: 25px;
    border-radius: 12px;
    margin-bottom: 10px;
    border: 1px solid #e0e0e0;
    box-shadow: 0px 4px 10px rgba(0,0,0,0.05);
}
.custom-divider {
    margin-top: 5px;
    margin-bottom: 20px;
    height: 2px;
    background: #e962f5;
}


st.markdown("<div class='card'>", unsafe_allow_html=True)
</style>

<div class="topnav">
  <a href="/app" class="active">Home</a>
  <a href="/Data_Cleaning">Data Cleaning</a>
  <a href="/Data_Visualization">Data Visualization</a>
  <a href="/Model_Training">Model_Training</a>
  <a href="/Post-Review">Project Review</a>
</div>
"""
st.markdown(navbar, unsafe_allow_html=True)

st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)

st.markdown("<div class='card'>", unsafe_allow_html=True)
st.title(" Data Cleaning & Preprocessing")
st.write("Prepare, clean, and explore your dataset with automated preprocessing steps.")


uploaded_files = st.file_uploader("Upload your CSV files", type=["csv"], accept_multiple_files=True)

@st.cache_data
def load_and_clean_data(files):
    if not files:
        return None, "Please upload at least one dataset."

    dfs = [pd.read_csv(file, parse_dates=["Date"], low_memory=False) for file in files]
    dataset = pd.concat(dfs, ignore_index=True)

  
    dataset.drop_duplicates(inplace=True)

   
    dataset["Date"] = pd.to_datetime(dataset["Date"], errors="coerce")
    dataset["Date"].fillna(dataset["Date"].median(), inplace=True)

   
    pollutants = ['PM2.5','PM10','NO','NO2','NOx','NH3','CO','SO2','O3','Benzene','Toluene','Xylene']
    for col in pollutants:
        if col in dataset.columns:
            dataset[col].fillna(dataset[col].median(), inplace=True)

    
    if "AQI" in dataset.columns:
        dataset["AQI"].fillna(dataset["AQI"].median(), inplace=True)

   
    def assign_bucket(aqi):
        if aqi <= 50: return "Good"
        elif aqi <= 100: return "Satisfactory"
        elif aqi <= 200: return "Moderate"
        elif aqi <= 300: return "Poor"
        elif aqi <= 400: return "Very Poor"
        else: return "Severe"

    if "AQI_Bucket" in dataset.columns:
        dataset["AQI_Bucket"] = dataset.apply(
            lambda row: assign_bucket(row["AQI"]) if pd.isna(row["AQI_Bucket"]) else row["AQI_Bucket"],
            axis=1
        )

    return dataset, f"Data successfully cleaned. Shape: {dataset.shape}"


if uploaded_files:
    data_frame, message = load_and_clean_data(uploaded_files)
    st.success(message)

    st.markdown("### 🔍 Preview of Cleaned Dataset")
    st.dataframe(data_frame.head(10), use_container_width=True)

    st.markdown("### 🧪 Missing Values After Cleaning")
    st.dataframe(data_frame.isnull().sum())

else:
    st.info("Upload one or more CSV files to begin cleaning.")


st.markdown("---")
st.subheader(" Load Data From Local Folder (Auto-Processing)")

folder_path = "Assessment Data-20251028"

@st.cache_data
def load_and_clean_data_from_folder(f_path):
    if not os.path.exists(f_path):
        return None, f"Folder not found at: {f_path}"

    files = glob(os.path.join(f_path, "*.csv"))
    if not files:
        return None, "No CSV files found in folder."

    dfs = [pd.read_csv(file, parse_dates=["Date"], low_memory=False) for file in files]
    dataset = pd.concat(dfs, ignore_index=True)

    dataset.drop_duplicates(inplace=True)

    dataset["Date"] = pd.to_datetime(dataset["Date"], errors="coerce")
    dataset["Date"].fillna(dataset["Date"].median(), inplace=True)

    pollutants = ['PM2.5','PM10','NO','NO2','NOx','NH3','CO','SO2','O3','Benzene','Toluene','Xylene']
    for c in pollutants:
        if c in dataset.columns:
            dataset[c].fillna(dataset[c].median(), inplace=True)

    if "AQI" in dataset.columns:
        dataset["AQI"].fillna(dataset["AQI"].median(), inplace=True)

    def assign_bucket(aqi):
        if aqi <= 50: return "Good"
        elif aqi <= 100: return "Satisfactory"
        elif aqi <= 200: return "Moderate"
        elif aqi <= 300: return "Poor"
        elif aqi <= 400: return "Very Poor"
        else: return "Severe"

    if "AQI_Bucket" in dataset.columns:
        dataset["AQI_Bucket"] = dataset.apply(
            lambda row: assign_bucket(row["AQI"]) if pd.isna(row["AQI_Bucket"]) else row["AQI_Bucket"],
            axis=1
        )

    return dataset, f"Folder data cleaned. Final shape: {dataset.shape}"

data_frame, msg = load_and_clean_data_from_folder(folder_path)

if data_frame is not None:
    st.success(msg)

    st.markdown("###  Combined Dataset Preview")
    st.dataframe(data_frame.head(15), use_container_width=True)

    st.markdown("---")
    st.subheader(" City-Wise Cleaning Results")

    for city in data_frame["City"].unique()[:6]:
        st.markdown(f"**City:** {city}")
        st.dataframe(data_frame[data_frame["City"] == city].head(6))
        st.markdown("---")

else:
    st.error(msg)


st.markdown("---")
st.header(" Exploratory Data Analysis (EDA)")

if data_frame is not None:

    st.subheader("Univariate Analysis")
    numeric_cols = data_frame.select_dtypes(include=[np.number]).columns
    categorical_cols = data_frame.select_dtypes(include=["object"]).columns

    col_select = st.selectbox("Select a column", list(numeric_cols) + list(categorical_cols))

    if col_select in numeric_cols:
        st.write(data_frame[col_select].describe())
        fig = px.histogram(data_frame, x=col_select, nbins=40)
        st.plotly_chart(fig, use_container_width=True)
    else:
        freq = data_frame[col_select].value_counts().reset_index()
        fig = px.bar(freq, x="index", y=col_select)
        st.plotly_chart(fig, use_container_width=True)

else:
    st.warning("Upload or load data to continue.")

st.markdown("---")
st.subheader("Bivariate Analysis")
st.write("Bivariate analysis examines the relationship between two variables.")

col_x = st.selectbox("Select X variable:", numeric_cols)
col_y = st.selectbox("Select Y variable:", numeric_cols)

st.write(f"### Relationship between **{col_x}** and **{col_y}**")
st.scatter_chart(data_frame[[col_x, col_y]])

correlation = data_frame[[col_x, col_y]].corr().iloc[0, 1]
st.info(f"Correlation between {col_x} and {col_y}: **{correlation:.3f}**")

st.markdown("---")

st.subheader("Multivariate Analysis")
st.write("Multivariate analysis explores more than two variables simultaneously.")




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




st.header("Missing Value Treatment")

if data_frame is None:
    st.error("No dataset available. Please upload or load data first.")
else:

    def missing_values_table(df):
        mis_val = df.isnull().sum()
        mis_val_percent = 100 * df.isnull().sum() / len(df)

        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
        mis_val_table_ren_columns = mis_val_table.rename(
            columns={0: 'Missing Values', 1: '% of Total Values'}
        )
        mis_val_table_ren_columns = mis_val_table_ren_columns[
            mis_val_table_ren_columns.iloc[:, 1] != 0
        ].sort_values('% of Total Values', ascending=False)

        return mis_val_table_ren_columns

    st.subheader("Missing Values (Before Cleaning)")
    st.dataframe(missing_values_table(data_frame))

    num_cols = ['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO',
                'SO2', 'O3', 'Benzene', 'Toluene', 'Xylene', 'AQI']

    cat_col = 'AQI_Bucket'

    if st.button("Clean Missing Values"):
        df1 = data_frame.copy()

        for col in num_cols:
            if col in df1.columns:
                if df1[col].isnull().any():
                    median_val = df1[col].median()
                    df1[col] = df1[col].fillna(median_val)

        if cat_col in df1.columns:
            if df1[cat_col].isnull().any():
                mode_val = df1[cat_col].mode()[0]
                df1[cat_col] = df1[cat_col].fillna(mode_val)

        st.success("Missing values have been successfully filled!")

        st.subheader("Missing Values (After Cleaning)")
        st.dataframe(missing_values_table(df1))

        st.session_state["clean_df"] = df1

    else:
        st.info("Click the button above to clean missing values.")

