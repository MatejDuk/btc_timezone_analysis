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
from histogram import Histogram

if "addresses" not in st.session_state:
    st.session_state.addresses = []

if "row" not in st.session_state:
    st.session_state.row = None

session = requests_cache.CachedSession('api_cache', expire_after=datetime.timedelta(days=30))

connection = pymysql.connect(
            host="gateway01.eu-central-1.prod.aws.tidbcloud.com",
            port=4000,
            user=os.getenv("USER_DATABASE"),
            password=os.getenv("PASSWORD_DATABASE"),
            database="test", 
            ssl={'ssl_verify_cert': False}
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
table_placeholder = st.empty()
if "first_address" in st.session_state and st.session_state.first_address is not None:
    table_placeholder.dataframe(st.session_state.first_address)

btc_address = st.text_input(
    label="Enter existing btc address"
)

if st.button("Start data collection and scraping"):   
    if btc_address:
        with st.spinner("Scraping blockchain data..."):
            

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
            clust_scrapper.heuristic_clus()

            st.session_state.addresses = clust_scrapper.addresses

            st.success("✅ Clustering successfully finished! All connected wallets have been scraped.")

    else:
        st.warning("Please provide a Bitcoin address first.")



st.write("---")
st.title("2. Data Visualisation")


if st.button("Generate Transaction Histogram"):
    # Check if we have addresses
    if st.session_state.addresses:
        with st.spinner("Analyzing transaction times..."):
            hist_gen = Histogram(connection, cursor)
            # Use the data from session_state
            fig, table = hist_gen.create_histogram(st.session_state.addresses)
            st.pyplot(fig)
            st.write(table)
            print(table.iloc[0].tolist())
            st.session_state.row = table.iloc[0].tolist()
    else:
        st.warning("No addresses found. Run Data Collection first!")


st.write("---")

st.title("3. Model Prediction")
# Input Section

if st.button("Calculate Probabilities"):
    # Wrap row inside an outer list to pass a 2D array matrix to scikit-learn
    y_row_proba = model.predict_proba([st.session_state.row])
    
    st.subheader("Predicted Time Zone Probabilities")
    
    # Turn the outputs into a legible dataframe mapping classes to probabilities
    if hasattr(model, "classes_"):
        prob_df = pd.DataFrame(y_row_proba, columns=model.classes_)
        st.dataframe(prob_df)
    else:
        st.write(y_row_proba)

print("Hello")
        
                
   



