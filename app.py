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

if "inputs" not in st.session_state:
    st.session_state.inputs = None

if "outputs" not in st.session_state:
    st.session_state.outputs = None

if "blockchain" not in st.session_state:
    st.session_state.blockchain = None

if "write" not in st.session_state:
    st.session_state.write = None

if "row" not in st.session_state:
    st.session_state.row = None

if "fig" not in st.session_state: 
    st.session_state.fig = None

if "model_row" not in st.session_state: 
    st.session_state.model_row = None

if "transactions_count" not in st.session_state:
    st.session_state.transactions_count = 0

# Cache session for saving API requests (expires after 30 days)
session = requests_cache.CachedSession('api_cache', expire_after=datetime.timedelta(days=30))

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
Welcome to the **Bitcoin Time Zone Predictor**. Developed as part of my Master's thesis, this application utilizes a machine learning model to classify a Bitcoin entity's geographic region—**Euro-Africa, Americas, East Asia & Pacific or Central Asia**—based entirely on its transaction history. 

The analysis pipeline is divided into three stages:

* **1. Data Collection & Clustering:** Input a valid Bitcoin address to initiate data extraction. The application automatically applies heuristic clustering to map the complete wallet architecture associated with the address. *(Note: This network analysis is computationally intensive; please allow time for the process to complete).*
* **2. Visualization & Vector Preparation:** Generates a chronological histogram of the combined transaction activity across all clustered addresses. This step also compiles the final input data row required for the model, which is made available for download.
* **3. Model Prediction:** Leverages a pre-trained XGBoost classification model to calculate the probability distribution across the four global regions.

---
*Disclaimer: Be aware that this is only hobby project and relies on free-tier APIs, which may occasionally result in rate limits or timeouts. If you encounter any bugs or wish to explore the underlying architecture, the complete repository is available on my GitHub. For direct inquiries, feel free to contact me at [matej.dukat@gmail.com](mailto:matej.dukat@gmail.com).*
""")
st.write("---")

st.title("1. Data collection")
st.info("""
**Address Requirements:** Please input a Bitcoin address known to belong to an **individual user**. 

Automated filtering of institutional or exchange wallets is beyond the scope of this project, so please verify the address entity manually using tools like [Arkham Intelligence](https://arkm.com/). 

*Want to test it out? Try one of these pre-verified addresses:*
* 18LHS5Guof1GHQgcXHQkHpKm4VV45Utnki
* 1N9vbvE2Yge7W2RVfDfvRuDVJXGQjPxz4D
* 1EmcSm2yyREEgkheJ2YUSyhaxkKno47NnK
* 15Yhg5Mj7tkheFi4yybpNFC3U4z6sQwRVb
""")
table_placeholder = st.empty()
if "write" in st.session_state and st.session_state.write is not None:
    table_placeholder.dataframe(st.session_state.write)

btc_address = st.text_input(
    label="Enter existing btc address"
)

if st.button("Start data collection and scraping"):   
    if btc_address:
        with st.spinner("Scraping starting address and connected transaction history..."):
            st.session_state.first_address = pd.DataFrame( 
                columns=["Address", "Number of outgoing txs", "Number of incoming txs", "Address source", "Iteration"]
            )
            table_placeholder.dataframe(st.session_state.first_address)
            st.session_state.write = None

            heur = HeuristicClustering(btc_address, session, table_placeholder)

            heur.heuristic_clus()

            st.session_state.addresses = heur.old_addresses
            st.session_state.inputs = heur.inputs_final
            st.session_state.outputs = heur.outputs_final
            st.session_state.blockchain = heur.blockchain_final
            
            st.success("✅ Clustering successfully finished! All connected wallets have been scraped.")

    else:
        st.warning("Please provide a Bitcoin address first.")



st.write("---")
st.title("2. Data Visualisation")

if st.button("Generate Transaction Histogram"):
    if st.session_state.addresses:
        with st.status("Analyzing transactions...", expanded=True) as status:
            st.write("Initializing Histogram class...")
            hist_gen = Histogram(st.session_state.addresses, st.session_state.inputs, st.session_state.outputs, st.session_state.blockchain)
            
            st.write("Calculating transaction distribution...")
            fig, table, trans = hist_gen.create_histogram(st.session_state.addresses)
            
            # Save to session state so they persist across reruns
            st.session_state.fig = fig
            st.session_state.model_row = table
            st.session_state.transactions_count = trans
            
            status.update(label="✅ Analysis complete!", state="complete", expanded=False)
    else:
        st.warning("No addresses found. Run Data Collection first!")

# Always display the histogram if it exists in state
if st.session_state.fig:
    st.pyplot(st.session_state.fig)
    st.dataframe(st.session_state.model_row)
    st.write(f"Model Input Vector Prepared ({st.session_state.transactions_count} transactions). You can download the data")

st.write("---")
st.title("3. Model Prediction")

if st.session_state.model_row is None:
    st.info("ℹ️ Please complete Data Collection (Step 1) and Generate Histogram (Step 2) to unlock predictions.")

# The button is disabled unless the histogram model_row is ready
if st.button("Calculate Probabilities", disabled=st.session_state.model_row is None):
    with st.spinner("Running machine learning model..."):
        # Predict class confidence probabilities
        probs = model.predict_proba([st.session_state.model_row.iloc[0].tolist()])[0]
        
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