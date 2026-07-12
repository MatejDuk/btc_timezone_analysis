import streamlit as st
import pickle
import pandas as pd
import numpy as np
import os
import requests_cache
import pymysql
import datetime
from get_address_info import GetAddressInfo
from heuristic_clustering import HeuristicClustering

session = requests_cache.CachedSession('api_cache', expire_after=datetime.timedelta(days=30))

connection = pymysql.connect(
            host="gateway01.eu-central-1.prod.aws.tidbcloud.com",
            port=4000,
            user=os.getenv("USER_DATABASE"),
            password=os.getenv("PASSWORD_DATABASE"),
            database="test", 
            ssl_verify_cert=True,
            ssl_verify_identity=True
        )
cursor = connection.cursor()

a = 0

# Set up page configurations
st.set_page_config(
    page_title="BTC Time Zone Predictor",
    page_icon="🌍",
    layout="centered"
)


# Load the trained model into memory securely
@st.cache_resource
def load_model():
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
btc_address = st.text_input(
    label="Enter existing btc address"
)

if st.button("Start data collection and scraping"):   
    if btc_address:
        with st.spinner("Scraping blockchain data..."):
            table_placeholder = st.empty()

            new_address = GetAddressInfo(btc_address, a, connection, cursor, session)
            # Save the returned proxy pointer directly into session state
            new_address.address_write()
            a = new_address.a
            
            st.session_state.first_address = pd.DataFrame(
                [[new_address.address, new_address.outgoing_count, new_address.incoming_count, "Inserted address", 0]], 
                columns=["Address", "Number of outgoing txs", "Number of incoming txs", "Address source", "Iteration"]
            )
            table_placeholder.dataframe(st.session_state.first_address)

            clust_scrapper = HeuristicClustering(btc_address, a, connection, cursor, session, table_placeholder)
            addresses = clust_scrapper.heuristic_clus()

            st.success("✅ Clustering successfully finished! All connected wallets have been scraped.")
            st.write(addresses)
    else:
        st.warning("Please provide a Bitcoin address first.")

st.write("---")
st.title("2. Data Visualisation")



st.write("---")

st.title("3. Model Prediction")
# Input Section
user_input = st.text_area(
    label="Enter a data row (comma-separated 24-hour vector ratios):",
    placeholder="0.01, 0.05, 0.12, 0.02, ...",
    help="Provide the exact 24 numerical features corresponding to your transaction time distribution.",
    height=150
)

if st.button("Calculate Probabilities"):
    if user_input.strip():
        try:
            # Strip outer brackets cleanly if the user pasted a raw Python list format
            clean_text = user_input.replace("[", "").replace("]", "")
            row = [float(val.strip()) for val in clean_text.split(",") if val.strip()]
            
            if model is not None:
                # Wrap row inside an outer list to pass a 2D array matrix to scikit-learn
                y_row_proba = model.predict_proba([row])
                
                st.subheader("Predicted Time Zone Probabilities")
                
                # Turn the outputs into a legible dataframe mapping classes to probabilities
                if hasattr(model, "classes_"):
                    prob_df = pd.DataFrame(y_row_proba, columns=model.classes_)
                    st.dataframe(prob_df)
                else:
                    st.write(y_row_proba)
            else:
                st.error("Model is not loaded.")
                
        except ValueError as e:
            st.error(f"Data parsing failed: Ensure the row contains only numerical values. Details: {e}")
    else:
        st.warning("Please input a feature row vector before evaluating prediction probabilities.")



