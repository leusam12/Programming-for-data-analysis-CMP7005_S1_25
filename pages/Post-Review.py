import streamlit as st

st.set_page_config(page_title="Machine Learning")

st.title('Data Output Prediction')
st.write('View model predictions and metrics here.')

st.title("Logistic Regression Evaluation")



st.write("""
The Logistic Regression model actually did a pretty good job, reaching about 96% accuracy. It was able to correctly classify most of the AQI categories, especially the major ones like Moderate, Poor, and Satisfactory. These categories had very strong precision and recall scores.

The only area where the model struggled a bit was with the Good category, where it mixed things up more often. This is not surprising because Logistic Regression is a simpler model and sometimes has trouble with complex boundaries between classes.

Overall, the model still performs well and gives a solid baseline, but it’s clear that more advanced models like Random Forest and XGBoost do a better job at capturing the full patterns in the data.
""")


st.title("Comparison Between Random Forest and Logistic Regression")



st.write("""
Both models did a solid job, but Random Forest clearly stood out. It achieved almost perfect accuracy, correctly predicting nearly every AQI category. Even the tricky categories like Very Poor and Severe were handled really well. This makes sense because Random Forest is great at picking up complex patterns in the data.

Logistic Regression also performed nicely with about 96% accuracy, but it struggled a bit more—especially with the Good category. This happens because Logistic Regression is a simpler model and doesn’t capture complicated relationships as well as Random Forest does.

**In simple terms:**
Random Forest understands the data better and gives much more accurate predictions.
Logistic Regression is good, but not as strong for this type of problem.
""")



