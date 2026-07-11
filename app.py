import streamlit as st
import pickle
import pandas as pd
import numpy as np
from get_address_info import GetAddressInfo

# Set up page configurations
st.set_page_config(
    page_title="BTC Time Zone Predictor",
    page_icon="🌍",
    layout="centered"
)

# Load the trained model into memory securely
@st.cache_resource
def load_model():
    # Looks for the model inside the models/ directory
    try:
        with open('timezone_model.pkl', 'rb') as file:
            return pickle.load(file)
    except FileNotFoundError:
        st.error("Model file 'timezone_model.pkl' not found. Please upload it to your repository.")
        return None

model = load_model()

# Header layout
st.title("🌍 Bitcoin Time Zone Predictor")
st.markdown("""
This application utilizes a pre-trained machine learning model to infer the primary 
operating time zone of a Bitcoin address based on its transactional behavioral features.
""")
st.write("---")

st.title("1. Data collection")
btc_address = st.text_area(
    label="Enter existing btc address",
    height = 20
)
if st.button("Start heuristic clustering and data collection."):   
    new_address = GetAddressInfo(btc_address,0)
    new_address.address_write()
st.write("---")

st.title("2. Model")
# Input Section
user_input = st.text_area(
    label="Enter a data row (comma-separated):",
    placeholder="Give a transaction history of specific BTC user in list format separated by commas [0.01, 0.01, ...]",
    help="Provide the exact numerical values in order, separated by commas.",
    height = 300
)

clean_text = user_input.replace("[", "").replace("]", "")

row = [float(val.strip()) for val in clean_text.split(",") if val.strip()]

if st.button("Calculate Probabilities"):
    y_row_proba = model.predict_proba([row])
    st.write(y_row_proba)



