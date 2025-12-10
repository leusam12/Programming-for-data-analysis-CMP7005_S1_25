import streamlit as st
import os

hide_sidebar = """
<style>
    [data-testid="stSidebar"] {display: none;}
    [data-testid="collapsedControl"] {display: none;}
</style>
"""
st.markdown(hide_sidebar, unsafe_allow_html=True)

navbar = """
<style>
.topnav {
    background-color: #1f2937;
    padding: 20px 30px;
    border-radius: 8px;
    width: 1000px;
    display: flex;
    justify-content: space-between;  
    align-items: center;
    margin-top:-60px;
    margin-left: 80px;
}

.topnav a {
    color: #f2f2f2;
    text-decoration: none;
    font-size: 17px;
    padding: 12px 25px;  
    border-radius: 5px;
    display: inline-block;
}

.topnav a:hover {
    background-color: #4b5563;
    color: white;
}

.topnav a.active {
    background-color: #2563eb;
    color: white;
}

.main {
    background-color: #2f7aeb;
}


.card {
    background: #e962f5;
    padding: 25px;
    border-radius: 12px;
    margin-bottom: 10px;
    border: 1px solid #e0e0e0;
    box-shadow: 0px 4px 10px rgba(0,0,0,0.05);
}


.section-title {
    font-size: 30px;
    color: #2c7be5;
    font-weight: 700;
    margin-bottom: 15px;
}


.subsection-title {
    font-size: 22px;
    color: #ff9900;
    font-weight: 600;
    margin-top: 5px;
}

.custom-divider {
    margin-top: 5px;
    margin-bottom: 20px;
    height: 2px;
    background: #e962f5;
}


.center-text {
    text-align: center;
}


ul li {
    margin-bottom: 5px;
}
h4 {
    text-align = center
    }
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


st.set_page_config(page_title="Overview", layout="wide")




st.markdown("<h1 class='center-text'> Air Quality Prediction  Overview</h1>", unsafe_allow_html=True)
st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)


st.markdown("<div class='card'>", unsafe_allow_html=True)
st.markdown("<h2 class='section-title'> Project Objective</h2>", unsafe_allow_html=True)

st.write("""
This project analyzes daily air quality data collected from major Indian cities between **2015 and 2020**.  
The dataset includes key pollutants such as **PM2.5, PM10, NO, NO2, NOx, NH3, CO, SO2, O3**, and VOCs  
(**Benzene, Toluene, Xylene**), along with the **Air Quality Index (AQI)** and the categorical **AQI_Bucket**.

These measurements help capture pollution levels, seasonal patterns, and environmental conditions across regions.

""")
col1, col2, = st.columns(2)

with col1:
    st.markdown("<h4 class='subsection-title'> Project Goals:</h4>", unsafe_allow_html=True)
    st.write("""


- Clean and preprocess the dataset  
- Perform exploratory data analysis (EDA)  
- Engineer meaningful features  
- Build ML models to predict **AQI_Bucket**  
- Develop a GUI for interactive data exploration & prediction  
- Document the workflow on GitHub  


""")

with col2:
    st.markdown("<h4 class='subsection-title'>Important of this project</h4>", unsafe_allow_html=True)
    st.write("""
This project is important because it provides a complete system for understanding, visualizing, and predicting air quality using real environmental data. 
It helps identify pollution patterns, highlights dangerous pollutant levels, and supports better decision-making for public health and environmental management. By combining data analysis, 
feature engineering, machine learning models, and an interactive Streamlit dashboard, the project makes it easy for users to explore insights, compare cities, and predict AQI categories in real-time.
This helps reveal pollutant behaviors, relationships, and insights for environmental monitoring.
""")

st.markdown("</div>", unsafe_allow_html=True)



st.markdown("<div class='card'>", unsafe_allow_html=True)
st.markdown("<h2 class='section-title'> Exploratory Data Analysis (EDA) Techniques</h2>", unsafe_allow_html=True)

st.write("""
To properly understand the dataset, three major analysis types were used:
""")


col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("<h4 class='subsection-title'>1. Univariate Analysis</h4>", unsafe_allow_html=True)
    st.write("""
Univariate analysis examines **one variable at a time**, helping understand:
- PM2.5 histograms  
- Summary statistics  
- AQI category frequency  

Useful for spotting outliers and skewness.
""")

with col2:
    st.markdown("<h4 class='subsection-title'>2. Bivariate Analysis</h4>", unsafe_allow_html=True)
    st.write("""
Bivariate analysis explores relationships between **two variables**, such as:
- PM2.5 vs AQI scatter  
- NO2 vs NOx correlation  
- AQI buckets across cities  

Reveals pollutant–AQI influence.
""")

with col3:
    st.markdown("<h4 class='subsection-title'>3. Multivariate Analysis</h4>", unsafe_allow_html=True)
    st.write("""
Multivariate analysis examines **3+ variables**, including:
- Correlation matrix  
- Heatmaps  
- Combined pollutant trends  

Shows deeper pollutant interactions.  
During data exploration, missing values were visualized to identify patterns before cleaning.
""")

st.markdown("</div>", unsafe_allow_html=True)



st.markdown("<div class='card'>", unsafe_allow_html=True)
st.markdown("<h2 class='section-title'> Static Analysis Plots</h2>", unsafe_allow_html=True)


col1, col2, = st.columns(2)

with col1:
    st.markdown("<h4 class='subsection-title'>The heatmap</h4>", unsafe_allow_html=True)
    st.write("""
The heatmap highlights which columns contain missing values and their distribution.  
Missing readings often come from malfunctioning sensors, maintenance periods, or incomplete reporting.  
This justified using **median-based imputation** during data cleaning.This helps reveal combined pollutant interactions and deeper environmental insights. In the data exploration stage, 
""")

with col2:
    st.markdown("<h4 class='subsection-title'>Deeper Environmental Insights</h4>", unsafe_allow_html=True)
    st.write("""
I took time to understand what the combined dataset actually looks like and how reliable it is. I reviewed its structure, checked the basic statistics of the numerical features, and looked at how each pollutant varies. 
  I also examined missing values to see which columns had gaps, and used a heatmap to visually highlight where those missing entries appeared. This step helped me identify the parts of the dataset that needed cleaning or extra attention before moving 
  into deeper analysis and modelling.
""")



image_folder = "images"
missing_heatmap = os.path.join(image_folder, "missing_values_heatmap.png")
correlation_heatmap = os.path.join(image_folder, "correlation_heatmap.png")
pollutant_distribution = os.path.join(image_folder, "pollutant_distribution.png")


colA, colB = st.columns(2)

with colA:
    if os.path.exists(missing_heatmap):
        st.image(missing_heatmap, caption="Missing Values Heatmap", use_column_width=True)
    else:
        st.warning("Missing values heatmap not found!")

    if os.path.exists(correlation_heatmap):
        st.image(correlation_heatmap, caption="Correlation Heatmap", use_column_width=True)
    else:
        st.warning("Correlation heatmap not found!")

with colB:
    if os.path.exists(pollutant_distribution):
        st.image(pollutant_distribution, caption="Pollutant Distribution", use_column_width=True)
    else:
        st.warning("Pollutant distribution plot not found!")

st.markdown("</div>", unsafe_allow_html=True)
