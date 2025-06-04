import joblib
import streamlit as st
import numpy as np


model = joblib.load('tips/tips.pkl')

total_bill = st.number_input('Enter total bill')
tip = st.number_input('Enter tip')
size = st.number_input('enter size')

if st.button('Classify'):
    input_features = np.array([[total_bill, tip, size]])
    pred = model.predict(input_features)
    mapping = ['Male', 'Female']
    st.success(f'The prediction is {pred}{mapping}[0]') 



