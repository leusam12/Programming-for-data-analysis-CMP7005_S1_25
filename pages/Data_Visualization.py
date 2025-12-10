import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import os
from glob import glob
import plotly.graph_objects as go 

# ------------------------------------------------
# HIDE SIDEBAR
# ------------------------------------------------
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

body {
    background-color: #f5f6fa;
}

.custom-divider {
    margin-top: 5px;
    margin-bottom: 20px;
    height: 2px;
    background: #e962f5;
}

.card {
    background: #e962f5;
    padding: 25px;
    border-radius: 12px;
    margin-bottom: 10px;
    border: 1px solid #e0e0e0;
    box-shadow: 0px 4px 10px rgba(0,0,0,0.05);
}
</style>

<div class="topnav">
  <a href="/app">Home</a>
  <a href="/Data_Cleaning">Data Cleaning</a>
  <a href="/Data_Visualization" class="active">Data Visualization</a>
  <a href="/Model_Training">Model Training</a>
  <a href="/Post-Review">Project Review</a>
</div>
"""
st.markdown(navbar, unsafe_allow_html=True)



st.set_page_config(page_title="Data Visualization", layout="wide")
st.markdown("<h1 style='text-align:center;'> Data Visualization & Insights </h1>", unsafe_allow_html=True)
st.markdown("<hr style='border: 1px solid #e962f5;'>", unsafe_allow_html=True)

st.markdown("<div class='card'>", unsafe_allow_html=True)


folder_path = "Assessment Data-20251028"

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

    dataset = dataset.loc[:, ~dataset.columns.duplicated()]
    dataset.columns = dataset.columns.map(str)
   
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

    dataset['AQI_Bucket'] = dataset.apply(
        lambda row: assign_bucket(row['AQI']) if pd.isna(row['AQI_Bucket']) else row['AQI_Bucket'], 
        axis=1
    )

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
    dataset['PM25_7day_avg'] = dataset.groupby('City')['PM2.5'].transform(lambda x: x.rolling(7, min_periods=1).mean())
    dataset['PM10_7day_avg'] = dataset.groupby('City')['PM10'].transform(lambda x: x.rolling(7, min_periods=1).mean())

    dataset['PM25_lag1'] = dataset.groupby('City')['PM2.5'].shift(1)
    dataset['PM10_lag1'] = dataset.groupby('City')['PM10'].shift(1)

    dataset['PM25_lag1'].fillna(dataset['PM25_lag1'].median(), inplace=True)
    dataset['PM10_lag1'].fillna(dataset['PM10_lag1'].median(), inplace=True)

    dataset['City_Code'] = dataset['City'].astype('category').cat.codes
    dataset['Season_Code'] = dataset['Season'].astype('category').cat.codes

    return dataset


df_engineered = load_clean_and_engineer_data(folder_path)



st.subheader("Data Engineered Features ")
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
st.subheader("Interactive Trend Plot (Plotly in precise)")
city_selection = st.selectbox("Select a City to Visualize Trends:", df_engineered['City'].unique())
if city_selection:
    city_data = df_engineered[df_engineered['City'] == city_selection]
    fig_px = px.line(city_data, x='Date', y='PM2.5', title=f"Daily PM2.5 Trends in {city_selection}")
    st.plotly_chart(fig_px, use_container_width=True)


st.subheader(" AQI Categories")

aqi_counts = df_engineered['AQI_Bucket'].value_counts().reset_index()


aqi_counts.columns = ['AQI_Bucket', 'Count']

fig_aqi = px.bar( 
    aqi_counts,
    x='AQI_Bucket',
    y='Count',
    title="Distribution of AQI Categories",
    template="plotly_white",
    color='AQI_Bucket'
)

st.plotly_chart(fig_aqi, use_container_width=True)

st.subheader("PM2.5 against AQI Relationship")


fig = px.scatter(
    df_engineered,  
    x="PM2.5",
    y="AQI",
    opacity=0.5,
    title="PM2.5 against AQI Relationship",
    template="plotly_white",
    trendline="ols"
)

st.plotly_chart(fig, use_container_width=True) 

st.subheader("Average concentration of pollutants")


pollutants = ['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3', 'Benzene', 'Toluene', 'Xylene']


pollutant_means = df_engineered[pollutants].mean().reset_index()
pollutant_means.columns = ["Pollutant", "Average"]


fig = px.bar(
    pollutant_means, 
    x="Pollutant",
    y="Average",
    title="Average Concentration of Each Pollutant",
    template="plotly_white",
    color="Pollutant"
)

st.plotly_chart(fig, use_container_width=True) 


st.subheader("Correlation heatmap for key pollutants (MULTIVARIATE ANALYSIS)")

pollutants = ['PM2.5','PM10','NO','NO2','NOx','NH3','CO','SO2','O3',
              'Benzene','Toluene','Xylene']


corr_matrix = df_engineered[pollutants].corr()


fig = go.Figure(
    data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.index,
        colorscale="RdBu",
        zmin=-1,
        zmax=1
    )
)

fig.update_layout(
    title="Correlation Heatmap of Pollutants",
    template="plotly_white"
)

st.plotly_chart(fig, use_container_width=True)


st.subheader("Monthly PM2.5 average trend")


monthly_avg = df_engineered.groupby('Month')['PM2.5'].mean().reset_index()

fig = px.line(
    monthly_avg,
    x='Month',
    y='PM2.5',
    title="Average PM2.5 Levels by Month",
    markers=True,
    template="plotly_white"
)

st.plotly_chart(fig, use_container_width=True)


st.subheader("AQI distribution by season and city")


aqi_sunburst = df_engineered.groupby(["Season", "City"])["AQI"].mean().reset_index()


fig = px.sunburst(
    aqi_sunburst,
    path=["Season", "City"],
    values="AQI",
    title="AQI Distribution by Season and City"
)


st.plotly_chart(fig, use_container_width=True)

st.subheader("PM2.5 variation across cities")

dataset = df_engineered.groupby('City')['PM2.5'].mean().reset_index()
fig = px.box(
    dataset,
    x="City",
    y="PM2.5",
    title="PM2.5 Levels Across Cities",
    template="plotly_white"
)
fig.update_layout(xaxis={'categoryorder': 'total descending'})

st.plotly_chart(fig, use_container_width=True)

pollutant_cols = ['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3',
                  'Benzene', 'Toluene', 'Xylene']

highest_pollutant_city = df_engineered.groupby("City")[pollutant_cols].sum()
highest_pollutant_city["Total_Pollutants"] = highest_pollutant_city.sum(axis=1)
highest_pollutant_city = highest_pollutant_city.sort_values("Total_Pollutants", ascending=False)

fig = px.bar(
    highest_pollutant_city,
    x=highest_pollutant_city.index,
    y="Total_Pollutants",
    title="Total Pollutant Levels Across Cities (Highest to Lowest)",
)

fig.update_layout(
    xaxis_title="City",
    yaxis_title="Total Pollutants (Sum of all pollutants)",
    height=600,
    margin=dict(l=40, r=40, t=60, b=120)
)

fig.update_xaxes(tickangle=90)

st.plotly_chart(fig, use_container_width=True)


st.markdown('---')
st.header("Exploratory Data Analysis (EDA) Features")


st.subheader("Univariate Analysis")
st.write("Explore one variable at a time. Choose numeric or categorical variables.")

numeric_cols = df_engineered.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = df_engineered.select_dtypes(include=['object', 'category']).columns.tolist()

uni_column = st.selectbox("Select a column for Univariate Analysis:", numeric_cols + categorical_cols)

if uni_column in numeric_cols:
    st.write(f" Summary Statistics for **{uni_column}**")
    st.write(df_engineered[uni_column].describe())
    
    st.write(f" Histogram / Distribution for **{uni_column}**")
    fig_uni = px.histogram(df_engineered, x=uni_column, nbins=30, title=f"Distribution of {uni_column}")
    st.plotly_chart(fig_uni, use_container_width=True)
else:
    st.write(f" Frequency Counts for **{uni_column}**")
    freq_df = df_engineered[uni_column].value_counts().reset_index()
    freq_df.columns = [uni_column, "Count"]
    st.write(freq_df)
    
    fig_uni_cat = px.bar(freq_df, x=uni_column, y="Count", title=f"Frequency of {uni_column}", template="plotly_white")
    st.plotly_chart(fig_uni_cat, use_container_width=True)

st.markdown('----')
st.subheader("Bivariate Analysis")
st.write("Explore relationships between two variables.")

col_x = st.selectbox("Select X variable:", numeric_cols, key="biv_x")
col_y = st.selectbox("Select Y variable:", numeric_cols, key="biv_y")


if col_x == col_y:
    st.error(" Please select different variables.")
else:
    st.write(f"### Scatter Plot: **{col_x}** vs **{col_y}**")
    fig_biv = px.scatter(
        df_engineered,
        x=col_x,
        y=col_y,
        opacity=0.6,
        trendline="ols",
        title=f"{col_x} vs {col_y} Relationship"
    )
    st.plotly_chart(fig_biv, use_container_width=True)

    correlation = df_engineered[[col_x, col_y]].corr().iloc[0, 1]
    st.info(f"Correlation between {col_x} and {col_y}: **{correlation:.3f}**")

st.markdown('---')

st.subheader(" Multivariate Analysis")
st.write("Explore correlations between three or more numeric variables.")

multi_cols = st.multiselect("Select 3 or more numeric variables:", numeric_cols)

if len(multi_cols) >= 3:
    corr_matrix = df_engineered[multi_cols].corr()
    
    st.write("Correlation Matrix")
    st.dataframe(corr_matrix)
    
    st.write("### Interactive Heatmap")
    fig_multi = px.imshow(
        corr_matrix,
        text_auto=True,
        color_continuous_scale="RdBu",
        zmin=-1, zmax=1,
        title="Multivariate Correlation Heatmap"
    )
    st.plotly_chart(fig_multi, use_container_width=True)
else:
    st.warning("Please select at least three variables for Multivariate Analysis.")

st.subheader("PM2.5 Levels Across Seasons")

st.write("""
This boxplot shows how **PM2.5 levels vary across different seasons**.  
Seasonal patterns can highlight how weather conditions influence pollution levels:

- **Higher PM2.5 in Winter** is common due to temperature inversion and low wind movement.  
- **Lower levels in Monsoon** may occur because rain helps remove pollutants from the air.  
- **Moderate levels in Summer and Post-Monsoon** show transitional effects.

Understanding seasonal variation is important for recognizing pollution cycles 
and improving AQI prediction models.
""")


fig_season = px.box(
    df_engineered,
    x="Season",
    y="PM2.5",
    color="Season",
    title="PM2.5 Levels Across Seasons",
    template="plotly_white"
)

st.plotly_chart(fig_season, use_container_width=True)

st.write("""
The visualizations helped reveal clear patterns in the air-quality data. We saw that most cities tend to fall into the Moderate or Satisfactory AQI categories, while cities like Delhi and Lucknow experience more Poor and Very Poor conditions. Seasonal trends also stood out, PM2.5 levels were lowest around mid-year and rose sharply during winter, which matches real-life pollution behavior in India.

The city comparison charts showed that some locations carry a much heavier pollution load than others. Correlation plots confirmed that pollutants like PM2.5 and PM10 often increase together, meaning they are influenced by similar conditions. Scatterplots also helped show how changes in pollutant ratios relate to AQI levels.

Overall, the visual exploration made the dataset much easier to understand and highlighted the key factors that affect air quality across cities and seasons.
""")         

