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

# Initialize session states
if "addresses" not in st.session_state:
    st.session_state.addresses = []

if "row" not in st.session_state:
    st.session_state.row = None

if "fig" not in st.session_state: 
    st.session_state.fig = None

if "model_row" not in st.session_state: 
    st.session_state.model_row = None

# Cache session for saving API requests (expires after 30 days)
session = requests_cache.CachedSession('api_cache', expire_after=datetime.timedelta(days=30))

# Connect to TiDB database
connection = pymysql.connect(
    host="gateway01.eu-central-1.prod.aws.tidbcloud.com",
    port=4000,
    user=os.getenv("USER_DATABASE"),
    password=os.getenv("PASSWORD_DATABASE"),
    database="test", 
    ssl={'ssl_verify_cert': False}
)
cursor = connection.cursor()

# Parse proxies for the initial address scraping
raw_ips = os.getenv("IP_ADDRESSES")
ip_addresses = [ip.strip() for ip in raw_ips.split(",")] if raw_ips else []
proxies_list = []

for line in ip_addresses:
    try:
        ip, port, user, pwd = line.split(":")
        proxies_list.append({
            "http": f"http://{user}:{pwd}@{ip}:{port}",
            "https": f"http://{user}:{pwd}@{ip}:{port}"
        })
    except ValueError:
        pass

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
            data = pickle.load(file)
            return data["model"], data["encoder"]
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

model, encoder = load_model()

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
        with st.spinner("Scraping starting address and connected transaction history..."):
            
            # 1. Initialize and run the starting address scraper (using in-memory parsing)
            new_address = GetAddressInfo(btc_address, a, proxies_list, session)
            new_address.fetch_and_extract()
            a = new_address.a  # Carry forward the updated proxy index
            
            # 2. Bulk-insert the starting address data so Heuristic Clustering can read it
            chunk_size = 2000
            
            while new_address.batch_blockchain_data:
                chunk = new_address.batch_blockchain_data[:chunk_size]
                sql = "INSERT IGNORE INTO blockchain_data (txid, num_inputs, num_outputs, fee, mempool_entry_time, block_height) VALUES (%s, %s, %s, %s, %s, %s)"
                cursor.executemany(sql, chunk)
                new_address.batch_blockchain_data = new_address.batch_blockchain_data[chunk_size:]
                connection.commit()

            while new_address.batch_tx_inputs:
                chunk = new_address.batch_tx_inputs[:chunk_size]
                sql = "INSERT IGNORE INTO tx_inputs (txid, input_order, address, value) VALUES (%s, %s, %s, %s)"
                cursor.executemany(sql, chunk)
                new_address.batch_tx_inputs = new_address.batch_tx_inputs[chunk_size:]
                connection.commit()

            while new_address.batch_tx_outputs:
                chunk = new_address.batch_tx_outputs[:chunk_size]
                sql = "INSERT IGNORE INTO tx_outputs (txid, output_order, address, value) VALUES (%s, %s, %s, %s)"
                cursor.executemany(sql, chunk)
                new_address.batch_tx_outputs = new_address.batch_tx_outputs[chunk_size:]
                connection.commit()

            # 3. Render the initial starting row in Streamlit
            st.session_state.first_address = pd.DataFrame(
                [[new_address.address, new_address.outgoing_count, new_address.incoming_count, "Inserted address", 0]], 
                columns=["Address", "Number of outgoing txs", "Number of incoming txs", "Address source", "Iteration"]
            )
            table_placeholder.dataframe(st.session_state.first_address)

            # 4. Trigger parallelized, bulk heuristic clustering
            clust_scrapper = HeuristicClustering(btc_address, a, connection, cursor, session, table_placeholder)
            clust_scrapper.heuristic_clus()

            st.session_state.addresses = clust_scrapper.addresses
            st.success("✅ Clustering successfully finished! All connected wallets have been scraped.")

    else:
        st.warning("Please provide a Bitcoin address first.")

st.write("---")
st.title("2. Data Visualisation")

if st.button("Generate Transaction Histogram"):
    if st.session_state.addresses:
        with st.status("Analyzing transactions...", expanded=True) as status:
            st.write("Initializing Histogram class...")
            hist_gen = Histogram(connection, cursor)
            
            st.write("Calculating transaction distribution...")
            fig, table = hist_gen.create_histogram(st.session_state.addresses)
            
            # Save to session state so they persist across reruns
            st.session_state.fig = fig
            st.session_state.model_row = table.iloc[0].tolist()
            
            status.update(label="✅ Analysis complete!", state="complete", expanded=False)
    else:
        st.warning("No addresses found. Run Data Collection first!")

# Always display the histogram if it exists in state
if st.session_state.fig:
    st.pyplot(st.session_state.fig)
    st.write("Model Input Vector Prepared.")

st.write("---")
st.title("3. Model Prediction")

if st.session_state.model_row is None:
    st.info("ℹ️ Please complete Data Collection (Step 1) and Generate Histogram (Step 2) to unlock predictions.")

# The button is disabled unless the histogram model_row is ready
if st.button("Calculate Probabilities", disabled=st.session_state.model_row is None):
    with st.spinner("Running machine learning model..."):
        # Predict class confidence probabilities
        probs = model.predict_proba([st.session_state.model_row])[0]
        
        # Resolve class names via label encoder or raw indexes
        if encoder:
            class_names = encoder.classes_
        else:
            class_names = [f"Class {i}" for i in range(len(probs))]
        
        # Prepare and sort data for visualization
        prob_df = pd.DataFrame({
            "Time Zone": class_names,
            "Confidence": probs
        }).sort_values(by="Confidence", ascending=False)
        
    st.subheader("Predicted Time Zone Probabilities")
    st.bar_chart(prob_df, x="Time Zone", y="Confidence", horizontal=True)
    st.dataframe(prob_df.set_index("Time Zone"))