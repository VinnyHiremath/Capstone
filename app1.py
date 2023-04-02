
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 22:07:42 2023

@author: VINNY
"""

base="dark"
backgroundColor="#291be6"
secondaryBackgroundColor="#959de8"

import streamlit as st
import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression

# Load the saved model
model = pickle.load(open('trained1_model.pkl', 'rb'))

with open('trained1_model.pkl', 'rb') as file:
    lr_model = pickle.load(file)
    
# create a Streamlit app
st.title('Predicting Customer Churn in a Telecom Company')

# Define the Streamlit app
def app():
    # Define the input fields for customer data
    gender = st.selectbox('Gender', ['Male', 'Female'])
    senior_citizen = st.selectbox('Senior Citizen', ['Yes', 'No'])
    partner = st.selectbox('Partner', ['Yes', 'No'])
    dependents = st.selectbox('Dependents', ['Yes', 'No'])
    tenure = st.number_input('Tenure')
    phone_service = st.selectbox('Phone Service', ['Yes', 'No'])
    multiple_lines = st.selectbox('Multiple Lines', ['Yes', 'No', 'No phone service'])
    internet_service = st.selectbox('Internet Service', ['DSL', 'Fiber optic', 'No'])
    online_security = st.selectbox('Online Security', ['Yes', 'No', 'No internet service'])
    online_backup = st.selectbox('Online Backup', ['Yes', 'No', 'No internet service'])
    device_protection = st.selectbox('Device Protection', ['Yes', 'No', 'No internet service'])
    tech_support = st.selectbox('Tech Support', ['Yes', 'No', 'No internet service'])
    streaming_tv = st.selectbox('Streaming TV', ['Yes', 'No', 'No internet service'])
    streaming_movies = st.selectbox('Streaming Movies', ['Yes', 'No', 'No internet service'])
    contract = st.selectbox('Contract', ['Month-to-month', 'One year', 'Two year'])
    paperless_billing = st.selectbox('Paperless Billing', ['Yes', 'No'])
    payment_method = st.selectbox('Payment Method', ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'])
    monthly_charges = st.number_input('Monthly Charges')
    total_charges = st.number_input('Total Charges')
    
    # Convert the input data into a format that can be used by the model
    data = pd.DataFrame([[gender, senior_citizen, partner, dependents, tenure, phone_service, multiple_lines, internet_service, online_security,
            online_backup, device_protection, tech_support, streaming_tv, streaming_movies, contract, paperless_billing, payment_method,
            monthly_charges, total_charges]], columns=['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 'PhoneService',
            'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
            'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod', 'MonthlyCharges', 'TotalCharges'])
    
    # One-hot encode the categorical features
    categorical_cols = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService',
                        'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
                        'Contract', 'PaperlessBilling', 'PaymentMethod']
    data_encoded = pd.get_dummies(data, columns=categorical_cols)
    
    # Make a prediction using the model
    churn_probability = model.predict_proba(data_encoded)[0, 1]
    y_pred = model.predict(data_encoded)
    
    
        
    # Return the churn prediction to the user
    if churn_probability > 0.5:
        st.write('This customer is likely to churn with a probability of ', churn_probability)
    else:
        st.write('This customer is not likely to churn with a probability of ', churn_probability)
    
    # make a prediction and display the result
    if st.button('Predict'):
        if y_pred == 1:
            st.success('Predicted Churn is : Yes')
        else:
            st.success('Predicted Churn is : No')
app() 


        
