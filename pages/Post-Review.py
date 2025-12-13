import streamlit as st

st.set_page_config(page_title="Machine Learning", layout="wide")hide_sidebar = """
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
  <a href="/Data_Visualization">Data Visualization</a>
  <a href="/Model_Training">Model Training</a>
  <a href="/Post-Review"  class="active">Project Review</a>
</div>
"""
st.markdown(navbar, unsafe_allow_html=True)

st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)

st.markdown("<div class='card'>", unsafe_allow_html=True)


st.sidebar.success("""This project predicts air quality (AQI and AQI categories) using multiple machine learning models. 
It includes feature engineering, model training, evaluation, and comparison of Logistic Regression, Decision Tree, and Random Forest models.""")

st.title('Data Output Prediction')
st.write('Explore model predictions, evaluation metrics, and insights from different machine learning models on AQI data.')

st.markdown('---')
st.header("Random Forest Classifier Evaluation")

st.write("""
The Random Forest model is the main classifier used to predict the AQI categories. It performed exceptionally well, achieving near-perfect accuracy on the test set. 
Random Forest is an ensemble method that builds multiple decision trees and averages their predictions, which allows it to handle complex relationships in the data effectively.

Key observations:
- The model correctly predicted almost every AQI category, including the challenging ones like Very Poor and Severe.
- Particulate matter (PM2.5 and PM10) and AQI itself were the most influential features, as shown in the feature importance analysis.
- The prediction confidence distribution shows that the model is highly confident for most predictions.
Overall, Random Forest provides a robust and reliable classification baseline for AQI categories.
""")

st.markdown('---')
st.header("Decision Tree Classifier Evaluation")

st.write("""
The Decision Tree model was added as a simpler alternative to Random Forest. It creates a tree structure where decisions are made based on feature thresholds.

Key points:
- Decision Trees are easy to interpret and can show exactly how a prediction was made.
- Although simpler than Random Forest, this model still performs reasonably well but may overfit on some categories.
- The confusion matrix and classification report help visualize where the model makes correct and incorrect predictions.
This provides a good benchmark to compare against more complex models like Random Forest.
""")

st.markdown('---')
st.header("Logistic Regression Evaluation")

st.write("""
The Logistic Regression model serves as a baseline linear classifier. Here’s how it performed:

- Achieved around 96% accuracy overall.
- Strong performance on major AQI categories like Moderate, Poor, and Satisfactory, with high precision and recall.
- Struggled a bit with the Good category, likely because simpler linear models can find it challenging to separate complex boundaries between classes.

Overall, Logistic Regression is a strong baseline and provides a solid understanding of the data, but more advanced models like Random Forest capture the full complexity more effectively.
""")

st.markdown('---')
st.header("Comparison: Random Forest vs Logistic Regression")

st.write("""
When comparing both models:

- **Random Forest** achieved near-perfect accuracy and handled all AQI categories well, including the challenging ones like Very Poor and Severe. It captures complex patterns and interactions between features effectively.
- **Logistic Regression** performed nicely with about 96% accuracy, but it struggled with simpler categories like Good, due to its linear nature.
**Summary:** Random Forest provides stronger predictive power, while Logistic Regression gives a clear baseline with simpler interpretability.
""")

st.markdown('---')
st.header("Regression Workflows for Predicting Numeric AQI")

st.write("""
In addition to classification, we also explored **regression models** to predict the numeric AQI values:

1. **Univariate Linear Regression:** 
   - Uses a single predictor to model AQI.
   - Simple and interpretable, but limited in capturing complex relationships.
   
2. **Bivariate Linear Regression:**
   - Uses two predictors for AQI prediction.
   - Provides slightly better accuracy than univariate regression, but still struggles with multi-feature interactions.
   
3. **Multivariate Random Forest Regression:**
   - Uses multiple predictors for AQI prediction.
   - Captures complex relationships and trends in the data.
   - Provides feature importances to understand which factors most influence AQI predictions.
   - Achieves high accuracy and low error metrics (RMSE, MAE), making it ideal for predicting numeric AQI.
""")

st.markdown('---')
st.header("Summary and Insights from All Models")

st.write("""
- **Random Forest (Classifier & Regressor)**: Most robust and accurate for both classification and numeric prediction.
- **Decision Tree**: Simple and interpretable; provides insight into decision paths.
- **Logistic Regression**: Solid baseline for classification; easy to interpret but limited for complex patterns.
- **Linear Regression (Uni/Bivariate)**: Useful for understanding relationships between AQI and individual or paired features, but limited for multi-feature interactions.
- **Feature Engineering** (e.g., PM2.5 & PM10 7-day averages, total VOC/Nitrogen, lag features) significantly improves model performance across all workflows.

Overall, combining feature engineering with advanced models like Random Forest gives the most reliable predictions, while simpler models help with interpretability and benchmarking.
""")
