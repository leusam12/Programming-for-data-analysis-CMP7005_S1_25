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

st.markdown('---')

st.subheader("Model Training and Results")

# The code runs automatically now, no 'if st.button' required
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
model = RandomForestClassifier(random_state=42)
full_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                ('model', model)])

with st.spinner('Training model... this might take a moment...'):
    full_pipeline.fit(X_train, y_train)
    y_pred = full_pipeline.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    st.success(f"Model Training Complete! Accuracy: {accuracy:.4f}")
    
    st.subheader("Model Evaluation Metrics")
    
    st.text("Classification Report:")
    report = classification_report(y_test, y_pred, output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose())

    st.text("Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred, labels=y.unique())
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=y.unique(), yticklabels=y.unique(), ax=ax)
    ax.set_ylabel('Actual')
    ax.set_xlabel('Predicted')
    st.pyplot(fig)



st.markdown('---')



st.subheader("Hyperparameter Tuning and Results")
st.write("""To further optimize the model, we use a grid search to find the best parameters. The results are displayed below automatically.""")


param_grid = {
    'model__n_estimators': [200, 300], # Added the list of values
    'model__max_depth': [10, 20]      # Added the list of values
}


grid_search = GridSearchCV(
    full_pipeline, 
    param_grid,
    cv=3,
    n_jobs=-1,
    verbose=1
)
st.write("**Grid Search Configuration:**")
st.write(grid_search) # This line generates the detailed text/visual output

with st.spinner('Running Grid Search automatically... This might take a while...'):
    grid_search.fit(X_train, y_train)
    st.success("Grid Search Complete!")
    st.write(f"Best parameters found: **{grid_search.best_params_}**")
    st.write(f"Best cross-validation score: **{grid_search.best_score_:.4f}**")
    
    # Display the best estimator configuration (matches your requested format)
    st.write("**Best Estimator Details:**")
    st.write(grid_search.best_estimator_) 

    results_df = pd.DataFrame(grid_search.cv_results_)
    st.dataframe(results_df[['param_model__n_estimators', 'param_model__max_depth', 'mean_test_score', 'rank_test_score']])

    


st.markdown('---')


st.subheader("Feature Importance Analysis (Top 15)")


ohe_features = grid_search.best_estimator_.named_steps['preprocessor'] \
    .named_transformers_['cat'].get_feature_names_out(categorical_cols)

all_features = np.concatenate([ohe_features, numerical_cols])

importances = grid_search.best_estimator_.named_steps['model'].feature_importances_
imp_df = pd.DataFrame({'Feature': all_features, 'Importance': importances})

top15 = imp_df.sort_values(by='Importance', ascending=False).head(15)


fig, ax = plt.subplots(figsize=(10,6))
sns.barplot(x=top15['Importance'], y=top15['Feature'], ax=ax)
plt.title("Top 15 Most Important Features")

st.pyplot(fig)
st.write("""
This chart shows the top 15 features that the Random Forest model relied on the most when predicting the AQI_Bucket. As expected, **AQI itself is the strongest predictor** because AQI_Bucket is directly derived from AQI ranges. After AQI, the next most important features are PM2.5, PM_Load, and the lag values like PM25_lag1, which capture how previous pollution levels influence current air quality.

Other pollutants such as PM10, CO, NO, NO2, SO2, and O3 also contribute meaningfully, though they have lower importance compared to the top variables. The presence of 7-day averages (like PM25_7day_avg) shows that the model benefits from understanding recent pollution trends, not just individual daily values.

Overall, this chart highlights that particulate matter (PM2.5 and PM10), AQI, and recent historical pollution levels are the most influential factors in determining the final air-quality category.
""")

st.markdown('---')




st.subheader("Random Forest Prediction Confidence Distribution")

rf_proba = full_pipeline.predict_proba(X_test)

fig, ax = plt.subplots(figsize=(10,6))
sns.histplot(rf_proba.max(axis=1), bins=30, kde=True, color='green', ax=ax)
plt.title("Random Forest Prediction Confidence Distribution")
plt.xlabel("Highest Predicted Probability")
plt.ylabel("Count")

# Use st.pyplot() instead of plt.show()
st.pyplot(fig)
st.write("""
This chart shows how confident the Random Forest model is when making predictions. Each bar represents the highest probability the model assigned to its chosen AQI category for a test sample. Most of the predictions fall toward the right side of the plot, close to 1.0, which means the model is highly confident about its decisions.

The smooth curve confirms that the majority of predictions have very high confidence scores.
""")


st.markdown('---')
st.header("MODEL 2: LOGISTIC REGRESSION")


logreg_pipeline = Pipeline(steps=[
    ('preprocess', preprocessor), 
    ('model', LogisticRegression(max_iter=3000))
])


with st.spinner('Training Logistic Regression model...'):
    logreg_pipeline.fit(X_train, y_train)
    log_preds = logreg_pipeline.predict(X_test)
    
   
    st.subheader("Logistic Regression Results")

    log_accuracy = accuracy_score(y_test, log_preds)
    st.write(f"**Logistic Regression Accuracy:** **{log_accuracy:.4f}**")

    st.text("Classification Report:")
    log_report = classification_report(y_test, log_preds, output_dict=True)
    st.dataframe(pd.DataFrame(log_report).transpose())


