import streamlit as st
import pandas as pd
import numpy as np
import os
from glob import glob
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

st.set_page_config(page_title="Model Training", layout="wide")
st.title('Model Training and Evaluation')
st.sidebar.success("Welcome to the Machine Learning Page")

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
        dfs.append(pd.read_csv(file, parse_dates=['Date'], low_memory=False))

    dataset = pd.concat(dfs, ignore_index=True)
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

    dataset['AQI_Bucket'] = dataset.apply(lambda row: assign_bucket(row['AQI']) 
                                         if pd.isna(row['AQI_Bucket']) else row['AQI_Bucket'], axis=1)

    dataset['Year'] = dataset['Date'].dt.year
    dataset['Month'] = dataset['Date'].dt.month
    dataset['DayOfWeek'] = dataset['Date'].dt.dayofweek
    dataset['Week'] = dataset['Date'].dt.isocalendar().week.astype(int)

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

# Load data
df = load_clean_and_engineer_data(folder_path)

st.subheader("Data Preparation for Modeling")

# Define features (X) and target (y)
X = df.drop(columns=['AQI_Bucket', 'Date'])
y = df['AQI_Bucket']

st.write(f"Features shape: {X.shape}, Target shape: {y.shape}")
st.dataframe(X.head())

categorical_cols = X.select_dtypes(include=['object', 'category']).columns
numerical_cols  = X.select_dtypes(exclude=['object', 'category']).columns

st.write(f"Numerical columns: {list(numerical_cols)}")
st.write(f"Categorical columns: {list(categorical_cols)}")

st.markdown('---')
st.subheader("Building the Preprocessing Pipeline")

numerical_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ]
)
st.success("Preprocessing pipeline defined successfully.")
st.markdown('---')

st.subheader("Model Training and Results")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
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

st.subheader("Random Forest Prediction Confidence Distribution")
rf_proba = full_pipeline.predict_proba(X_test)
fig, ax = plt.subplots(figsize=(10,6))
sns.histplot(rf_proba.max(axis=1), bins=30, kde=True, color='green', ax=ax)
plt.title("Random Forest Prediction Confidence Distribution")
plt.xlabel("Highest Predicted Probability")
plt.ylabel("Count")
st.pyplot(fig)
st.markdown('---')

st.header("LOGISTIC REGRESSION")
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

st.markdown('---')
st.header("Regression Workflows — Univariate, Bivariate, Multivariate (RF)")
st.write("Use these to predict **AQI** (numeric). Choose predictors and evaluate models.")

if 'AQI' not in df.columns:
    st.error("AQI column not found in dataframe. Regression requires a numeric 'AQI' target.")
else:
    target = 'AQI'
    all_numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    exclude = [target, 'Year', 'Month', 'DayOfWeek', 'Week', 'City_Code', 'Season_Code']
    candidate_features = [c for c in all_numeric if c not in exclude]
    st.write(f"Available numeric predictors (sample): {candidate_features[:10]}")

    test_size = st.slider("Test set size (%):", 5, 50, 20) / 100.0
    random_state = st.number_input("Random state (seed):", value=42, step=1)

    st.subheader("1) Univariate Linear Regression")
    uni_feat = st.selectbox("Select single predictor (univariate):", candidate_features, key="uni_feat")
    if st.button("Run Univariate Regression", key="run_uni"):
        X_uni = df[[uni_feat]].values.reshape(-1, 1)
        y_uni = df[target].values
        X_train_u, X_test_u, y_train_u, y_test_u = train_test_split(X_uni, y_uni, test_size=test_size, random_state=int(random_state))
        model_uni = LinearRegression()
        model_uni.fit(X_train_u, y_train_u)
        y_pred_u = model_uni.predict(X_test_u)
        mse = mean_squared_error(y_test_u, y_pred_u)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test_u, y_pred_u)
        r2 = r2_score(y_test_u, y_pred_u)
        st.metric("R²", f"{r2:.4f}")
        st.metric("RMSE", f"{rmse:.3f}")
        st.metric("MAE", f"{mae:.3f}")

        df_plot = pd.DataFrame({uni_feat: X_test_u.flatten(), 'Actual': y_test_u, 'Predicted': y_pred_u})
        fig = px.scatter(df_plot, x=uni_feat, y='Actual', opacity=0.6, labels={'y':'AQI'},
                         title=f"Univariate: Actual vs {uni_feat} — Predicted overlay")
        fig.add_traces(px.scatter(df_plot, x=uni_feat, y='Predicted').data)
        st.plotly_chart(fig, use_container_width=True)
        st.write("Model coefficients:")
        st.write({"intercept": float(model_uni.intercept_), "coef": float(model_uni.coef_[0])})

    st.subheader("Bivariate Linear Regression")
    biv_cols = st.multiselect("Select exactly 2 predictors (bivariate):", candidate_features, default=candidate_features[:2], key="biv_feats")
    if len(biv_cols) != 2:
        st.info("Please select exactly 2 predictors for bivariate regression.")
    else:
        if st.button("Run Bivariate Regression", key="run_bi"):
            X_bi = df[biv_cols].values
            y_bi = df[target].values
            X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(X_bi, y_bi, test_size=test_size, random_state=int(random_state))
            model_bi = LinearRegression()
            model_bi.fit(X_train_b, y_train_b)
            y_pred_b = model_bi.predict(X_test_b)
            mse = mean_squared_error(y_test_b, y_pred_b)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test_b, y_pred_b)
            r2 = r2_score(y_test_b, y_pred_b)
            st.metric("R²", f"{r2:.4f}")
            st.metric("RMSE", f"{rmse:.3f}")
            st.metric("MAE", f"{mae:.3f}")
            df_plot = pd.DataFrame({biv_cols[0]: X_test_b[:,0], biv_cols[1]: X_test_b[:,1], 'Actual': y_test_b, 'Predicted': y_pred_b})
            fig = px.scatter(df_plot, x=biv_cols[0], y='Actual', color=biv_cols[1],
                             title=f"Bivariate: Actual AQI vs {biv_cols[0]} colored by {biv_cols[1]}")
            fig.add_traces(px.scatter(df_plot, x=biv_cols[0], y='Predicted', marker=dict(symbol='x')).data)
            st.plotly_chart(fig, use_container_width=True)
            st.write("Model coefficients:")
            st.write({"intercept": float(model_bi.intercept_), "coefs": list(map(float, model_bi.coef_))})

    st.subheader("3) Multivariate Random Forest Regression")
    multi_feats = st.multiselect("Select predictors for RF (3 or more):", candidate_features, default=candidate_features[:8], key="multi_feats")
    n_estimators = st.slider("n_estimators (trees):", 50, 1000, 200, step=50)
    max_depth = st.slider("max_depth (None=0):", 0, 50, 12, step=1)

    if st.button("Run Random Forest Regression", key="run_rf"):
        if len(multi_feats) < 3:
            st.error("Please select at least 3 predictors for the Random Forest model.")
        else:
            X_rf = df[multi_feats].values
            y_rf = df[target].values
            X_train_rf, X_test_rf, y_train_rf, y_test_rf = train_test_split(X_rf, y_rf, test_size=test_size, random_state=int(random_state))
            rf = RandomForestRegressor(
                n_estimators=int(n_estimators),
                max_depth=(None if max_depth==0 else int(max_depth)),
                random_state=int(random_state),
                n_jobs=-1
            )
            with st.spinner("Training Random Forest..."):
                rf.fit(X_train_rf, y_train_rf)
            y_pred_rf = rf.predict(X_test_rf)

            mse = mean_squared_error(y_test_rf, y_pred_rf)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test_rf, y_pred_rf)
            r2 = r2_score(y_test_rf, y_pred_rf)

            st.metric("R²", f"{r2:.4f}")
            st.metric("RMSE", f"{rmse:.3f}")
            st.metric("MAE", f"{mae:.3f}")

            df_plot = pd.DataFrame({'Actual': y_test_rf, 'Predicted': y_pred_rf})
            fig = px.scatter(df_plot, x='Actual', y='Predicted', trendline="ols", title="Random Forest: Actual vs Predicted AQI")
            st.plotly_chart(fig, use_container_width=True)

            importances = rf.feature_importances_
            fi_df = pd.DataFrame({'feature': multi_feats, 'importance': importances}).sort_values('importance', ascending=False)
            st.subheader("Feature Importances (RF)")
            fig_fi = px.bar(fi_df.head(20), x='importance', y='feature', orientation='h', title="Top feature importances")
            st.plotly_chart(fig_fi, use_container_width=True)

            st.write("Sample predictions (first 10):")
            st.dataframe(pd.DataFrame({'Actual': y_test_rf[:10], 'Predicted': y_pred_rf[:10]}))


st.markdown('---')
st.header("Decision Tree Classifier")
dt_model = DecisionTreeClassifier(random_state=42)
dt_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', dt_model)
])
with st.spinner("Training Decision Tree..."):
    dt_pipeline.fit(X_train, y_train)
    y_pred_dt = dt_pipeline.predict(X_test)

    acc_dt = accuracy_score(y_test, y_pred_dt)
    st.success(f"Decision Tree Test Accuracy: {acc_dt:.4f}")

    st.subheader("Classification Report (Decision Tree)")
    dt_report = classification_report(y_test, y_pred_dt, output_dict=True)
    st.dataframe(pd.DataFrame(dt_report).transpose())

    st.subheader("Confusion Matrix (Decision Tree)")
    cm_dt = confusion_matrix(y_test, y_pred_dt, labels=np.unique(y_test))
    fig, ax = plt.subplots(figsize=(7,5))
    sns.heatmap(cm_dt, annot=True, fmt='d', xticklabels=np.unique(y_test),
                yticklabels=np.unique(y_test), cmap='Blues', ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Decision Tree - Confusion Matrix")
    st.pyplot(fig)


y_pred_lr = logreg_pipeline.predict(X_test)
acc_lr = accuracy_score(y_test, y_pred_lr)

st.subheader("Logistic Regression Evaluation")
st.success(f"Test Accuracy: {acc_lr:.4f}")

st.text("Classification Report:")
report_lr = classification_report(y_test, y_pred_lr, output_dict=True, digits=4)
st.dataframe(pd.DataFrame(report_lr).transpose())

cm_lr = confusion_matrix(y_test, y_pred_lr, labels=np.unique(y_test))
fig, ax = plt.subplots(figsize=(7,5))
sns.heatmap(cm_lr, annot=True, fmt='d', xticklabels=np.unique(y_test),
            yticklabels=np.unique(y_test), cmap='Oranges', ax=ax)
ax.set_title('Logistic Regression - Confusion Matrix')
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
st.pyplot(fig)
