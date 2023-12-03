import streamlit as st
import pandas as pd
import pickle as pk
from sklearn.preprocessing import StandardScaler

# Load the model and scaler
lg_loaded = pk.load(open('lg.pk1', 'rb'))
scaler_loaded = pk.load(open('scaler.pk1', 'rb'))

st.header('Loan prediction App')

# Slider and selectbox inputs
n_de = st.slider("Choose no of dependents", 0, 3, key="slider_dependents")
g_s = st.selectbox('Choose Education', ['Graduate', 'Not Graduate'], key="select_education")
s = st.selectbox('Self Employed?', ['Yes', 'No'], key="select_self_employed")
a_i = st.slider("Choose Applicants income", 0, 51000, key="slider_income")
l_a = st.slider("Choose loan amount", 0, 51000, key="slider_loan_amount")
l_d = st.slider("Choose loan duration", 0, 360, key="slider_loan_duration")


# Convert categorical inputs to numerical
if g_s == 'Graduate':
    g_os = 0
else:
    g_os = 1

if s == 'No':
    s_e = 0
else:
    s_e = 1

# Prediction button
if st.button("Predict"):
    # Create a DataFrame with the input values
    p_d = pd.DataFrame([[n_de, g_os, s_e, a_i, l_a, l_d]],
                       columns=['Dependents', 'Education', 'Self_Employed', 'ApplicantIncome', 'LoanAmount', 'Loan_Amount_Term'])

    # Convert 'ApplicantIncome' and 'LoanAmount' to numeric
    p_d['ApplicantIncome'] = p_d['ApplicantIncome'].replace('[\$,]', '', regex=True).astype(float)
    p_d['LoanAmount'] = p_d['LoanAmount'].replace('[\$,]', '', regex=True).astype(float)

   

    # Ensure categorical encoding matches training
    if g_s == 'Graduate':
        p_d['Education'] = 0
    else:
        p_d['Education'] = 1

    if s == 'No':
        p_d['Self_Employed'] = 0
    else:
        p_d['Self_Employed'] = 1


    # Ensure feature order matches training
    feature_order = ['Dependents', 'Education', 'Self_Employed', 'ApplicantIncome', 'LoanAmount', 'Loan_Amount_Term']
    p_d = p_d[feature_order]

    

    # Transform the input data using the loaded scaler
    p_d_scaled = scaler_loaded.transform(p_d)

    

    # Make the prediction using the loaded model
    threshold = 0.51
    predict_proba = lg_loaded.predict_proba(p_d_scaled)[:, 1]
    predict = (predict_proba > threshold).astype(int)

    

    # Display the prediction result
    if predict[0] == 1:
        st.markdown('Loan is Approved')
    else:
        st.markdown('Loan is Rejected')