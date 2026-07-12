import streamlit as st
import pickle
import pandas as pd
import numpy as np
from get_address_info import GetAddressInfo
from heuristic_clustering import HeuristicClustering

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

if st.button("Start data collection for inserted address."):   
    if btc_address:
        with st.spinner("Scraping blockchain data..."):
            new_address = GetAddressInfo(btc_address, 0)
            # Save the returned proxy pointer directly into session state
            new_address.address_write()
            a = new_address.a
            
            st.session_state.first_address = pd.DataFrame(
                [[new_address.address, new_address.outgoing_count, new_address.incoming_count, "Inserted address"]], 
                columns=["Address", "Number of outgoing txs", "Number of incoming txs", "Address source"]
            )
    else:
        st.warning("Please provide a Bitcoin address first.")

# Persist the dataframe visual across page reruns

if st.button("Start heuristic clustering"):
    if btc_address:
        with st.spinner("Executing multi-input and change address heuristics..."):
            # Safely fetch the index from session state
            clustering_agent = HeuristicClustering(
                start_address=btc_address, 
                start_ip_index=a
            )
            clustered_list = clustering_agent.heuristic_clus()
            
            st.success(f"Clustering finished! Found {len(clustered_list)} unique addresses.")
            st.dataframe(pd.DataFrame(clustered_list, columns=["Clustered Addresses"]))
    else:
        st.warning("Please provide a Bitcoin address first.")

st.write("---")

st.title("2. Model Prediction")
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



