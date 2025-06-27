# app.py

import streamlit as st
import numpy as np
import joblib
import pandas as pd

@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
    cols = ['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age','Outcome']
    return pd.read_csv(url, names=cols)

df = load_data()

# Load model and scaler
model = joblib.load("diabetes_model.pkl")
scaler = joblib.load("scaler.pkl")

# Title
st.title("🩺 Diabetes Prediction App")

if st.checkbox("Show dataset summary"):
    st.dataframe(df.describe())

with st.expander("ℹ️ Learn how each feature affects diabetes"):
    st.markdown("""
    #### 📊 Feature Insights

    - **Pregnancies**  
      More pregnancies may slightly increase long-term diabetes risk, especially due to gestational diabetes.

    - **Glucose**  
      Higher blood sugar levels are the strongest predictor of diabetes. Fasting glucose ≥ 126 mg/dL is a clinical marker. (normal range is 70–140). Even in severe diabetes, glucose may rise to ~300–600 mg/dL.

    - **Blood Pressure (BP)**  
      High blood pressure can increase insulin resistance, a key factor in Type 2 diabetes.

    - **Skin Thickness**  
      Indicates body fat. Increased skin fold thickness may reflect obesity risk, linked to diabetes.

    - **Insulin**  
      Low insulin levels or resistance to insulin action are central causes of diabetes.

    - **BMI (Body Mass Index)**  
      Obesity is a major risk factor for Type 2 diabetes. BMI ≥ 30 is considered obese.

    - **Diabetes Pedigree Function (DPF)**  
      This reflects genetic risk. Higher DPF means stronger family history of diabetes.

    - **Age**  
      Risk increases significantly after age 45. Aging reduces insulin sensitivity.

    > Understanding these features helps users take action — through lifestyle, diet, or medical care.
    """)

# ✅ Input Fields
pregnancies = st.number_input("Pregnancies", min_value=0, max_value=17, value=0)
glucose = st.number_input("Glucose", min_value=0, max_value=200, value=120)
bp = st.number_input("Blood Pressure", min_value=0, max_value=122, value=80)
skin = st.number_input("Skin Thickness", min_value=0, max_value=99, value=20)
insulin = st.number_input("Insulin", min_value=0, max_value=846, value=100)
bmi = st.number_input("BMI", min_value=0.0, max_value=67.0, value=25.0)
dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=2.5, value=0.5)
age = st.number_input("Age", min_value=0, max_value=100, value=30)

# ✅ Predict Button
if st.button("Predict"):
    input_dict = {
        'Pregnancies': [pregnancies],
        'Glucose': [glucose],
        'BloodPressure': [bp],
        'SkinThickness': [skin],
        'Insulin': [insulin],
        'BMI': [bmi],
        'DiabetesPedigreeFunction': [dpf],
        'Age': [age]
    }
    input_df = pd.DataFrame(input_dict)
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)[0]
    result = "Diabetic" if prediction == 1 else "Not Diabetic"
    st.success(f"Prediction: {result}")

    # 🔍 Health tips based on input
    st.subheader("🩺 Health Tips")

    if bmi >= 30:
        st.warning("Your BMI is in the obese range. Consider consulting a doctor or nutritionist.")
    elif bmi >= 25:
        st.info("Your BMI is in the overweight range. A balanced diet and regular exercise are recommended.")

    if glucose >= 140:
        st.warning("High glucose level detected. This may indicate hyperglycemia.")

    if bp >= 90:
        st.warning("High blood pressure detected. Monitor your BP regularly.")

    if age >= 45:
        st.info("Age over 45 is a common risk factor for Type 2 Diabetes.")

    if dpf >= 1.0:
        st.info("You may have a higher genetic predisposition to diabetes.")
