import streamlit as st
import os

st.set_page_config(page_title="Overview", layout="wide")

st.title('Project Objective')



st.write("""
This project analyzes daily air quality data collected from major Indian cities between 2015 and 2020. The dataset includes key pollutants such as PM2.5, PM10, NO, NO2, NOx, NH3, CO, SO2, O3, and VOCs (Benzene, Toluene, Xylene), as well as the Air Quality Index (AQI) and categorized AQI_Bucket. These measurements help describe pollution levels, seasonal patterns, and overall environmental conditions across different regions in India.

The aim of this project is to clean and preprocess the dataset, perform exploratory data analysis (EDA), engineer new features, and build a machine learning model to predict AQI_Bucket. The workflow includes visualizing pollutant behavior, analyzing correlations, training a classification model, and developing a simple GUI application for data exploration and prediction. GitHub is used for version control and project documentation.
""")

st.subheader("Static Analysis Plots")


st.write("""
The heatmap shows which columns contain missing values and how they are distributed. Most pollutants had scattered missing points, which is common in environmental datasets due to sensor downtime or incomplete reporting. This visualization helped confirm the need for imputation during data cleaning.
""")

image_path_1 = os.path.join("images", "opk.png")
image_path_2 = os.path.join("images", "Screenshot.png")

if os.path.exists(image_path_1):
    st.image(image_path_1, caption="Correlation heatmap", use_column_width=True)
else:
    st.warning(f"Image not found at path: {image_path_1}")

if os.path.exists(image_path_2):
    st.image(image_path_2, caption="Team Connect", use_column_width=True)
else:
    st.warning(f"Image not found at path: {image_path_2}")
