import streamlit as st
import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.models import load_model

# Load saved model and preprocessing objects
model = load_model('model.h5')

with open('label_encoder_gender.pkl', 'rb') as f:
    label_encoder_gender = pickle.load(f)

with open('onehot_encoder_geo.pkl', 'rb') as f:
    onehot_encoder_geo = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Streamlit UI
st.title("Customer Churn Prediction")

credit_score = st.number_input("Credit Score", min_value=300, max_value=900, step=1)
gender = st.selectbox("Gender", ['Male', 'Female'])
age = st.number_input("Age", min_value=18, max_value=100, step=1)
tenure = st.number_input("Tenure (years)", min_value=0, max_value=20, step=1)
balance = st.number_input("Account Balance", step=100.0)
num_of_products = st.number_input("Number of Products", min_value=1, max_value=4, step=1)
has_cr_card = st.selectbox("Has Credit Card", [0, 1])
is_active_member = st.selectbox("Is Active Member", [0, 1])
estimated_salary = st.number_input("Estimated Salary", step=100.0)
geography = st.selectbox("Geography", ['France', 'Spain', 'Germany'])

# On button click
if st.button("Predict"):
    # Encode gender and geography
    gender_encoded = label_encoder_gender.transform([gender])[0]
    geo_encoded = onehot_encoder_geo.transform([[geography]])
    
    # Create input DataFrame
    input_data = pd.DataFrame({
        'CreditScore': [credit_score],
        'Gender': [gender_encoded],
        'Age': [age],
        'Tenure': [tenure],
        'Balance': [balance],
        'NumOfProducts': [num_of_products],
        'HasCrCard': [has_cr_card],
        'IsActiveMember': [is_active_member],
        'EstimatedSalary': [estimated_salary]
    })

    geo_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))
    input_data = pd.concat([input_data, geo_df], axis=1)

    # Scale input
    input_scaled = scaler.transform(input_data)

    # Predict
    prediction = model.predict(input_scaled)

    # Display result
    st.subheader("Churn Prediction Result")
    st.write("Churn Probability:", float(prediction[0][0]))
    
    if prediction[0][0] >= 0.5:
        st.warning("The customer is likely to churn.")
    else:
        st.success("The customer is likely to stay.")
