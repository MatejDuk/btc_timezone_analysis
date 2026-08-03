from multiprocessing.dummy import connection
import streamlit as st
import pickle
import pandas as pd
import numpy as np
import pymysql
import os
import requests_cache
import datetime
import matplotlib.pyplot as plt
from get_address_info import GetAddressInfo
from heuristic_clustering import HeuristicClustering

# Initialize session states
if "address" not in st.session_state:
    st.session_state.address = None

if "addresses" not in st.session_state:
    st.session_state.addresses = []

if "total_transactions" not in st.session_state:
    st.session_state.total_transactions = 0

if "write" not in st.session_state:
    st.session_state.write = None

if "row" not in st.session_state:
    st.session_state.row = None

if "fig" not in st.session_state: 
    st.session_state.fig = None

if "model_row" not in st.session_state: 
    st.session_state.model_row = None
    
  

#Connection to DB
@st.cache_resource
def get_db_connection():
    connection = pymysql.connect(
        host = "gateway01.eu-central-1.prod.aws.tidbcloud.com",
        port = 4000,
        user = os.getenv("USER_DATABASE"),
        password = os.getenv("PASSWORD_DATABASE"),
        database = "test", 
        ssl_verify_cert = True,
        ssl_verify_identity = True
    )

    cursor = connection.cursor()
    return connection

connection = get_db_connection()

# Cache session for saving API requests (expires after 30 days)
@st.cache_resource
def get_cached_session():
    return requests_cache.CachedSession('api_cache', expire_after=datetime.timedelta(days=30))

session = get_cached_session()

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

@st.cache_data
def create_histogram(model_row):
    values = model_row.iloc[0].tolist()
    hours = list(range(24))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(hours, values, width=1.0, align='edge', color='skyblue', edgecolor='black')
    ax.set_title("Bitcoin Transaction Activity", size=16)
    ax.set_xlabel("Hour of the Day", size=14)
    ax.set_ylabel("Number of transactions", size=14)
    ax.set_xticks(np.arange(24) + 0.5)
    ax.set_xticklabels(range(24))
    return fig

def reset_state():
    st.session_state.write = None
    st.session_state.btc_address_input = ""
    st.session_state.addresses = []
    st.session_state.total_transactions = 0
    st.session_state.row = None
    st.session_state.fig = None
    st.session_state.model_row = None

def reset_write_state():
    st.session_state.write = None

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
type_of_data_collection = st.selectbox(
    label="Select data collection method",
    options=["Heuristic clustering", "Data extraction from saved addresses"],
    on_change=reset_state
)
if type_of_data_collection == "Heuristic clustering":
    st.info("""
    **Address Requirements:** Please input a Bitcoin address known to belong to an **individual user**. 

    Automated filtering of institutional or exchange wallets is beyond the scope of this project, so please verify the address entity manually using tools like [Arkham Intelligence](https://arkm.com/). 

    *Want to test it out? Try one of these pre-verified addresses:*
    * 18LHS5Guof1GHQgcXHQkHpKm4VV45Utnki
    * 1N9vbvE2Yge7W2RVfDfvRuDVJXGQjPxz4D
    * 1EmcSm2yyREEgkheJ2YUSyhaxkKno47NnK
    * 15Yhg5Mj7tkheFi4yybpNFC3U4z6sQwRVb
    """)
    btc_address = st.text_input(
        label="Enter existing btc address",
        persist_state = "session",
        key = "btc_address_input",
        on_change=reset_write_state
    )
    heur_btn = st.button("Start Heuristic Clustering")
    table_placeholder = st.empty()
    if "write" in st.session_state and st.session_state.write is not None:
        table_placeholder.dataframe(st.session_state.write)

    

    if heur_btn:   
        if btc_address:
            start_time = datetime.datetime.now()
            with st.spinner("Scraping starting address and connected transaction history..."):
                st.session_state.first_address = pd.DataFrame( 
                    columns=["Address", "Number of outgoing txs", "Number of incoming txs", "Address source", "Iteration"]
                )
                table_placeholder.dataframe(st.session_state.first_address)
                st.session_state.write = None

                heur = HeuristicClustering(btc_address, session, table_placeholder)

                heur.heuristic_clus()

                heur.get_final_data()
                st.session_state.model_row = heur.model_df
                st.session_state.row = heur.hour_counts
                st.session_state.total_transactions = heur.total_transactions

                st.session_state.addresses = heur.old_addresses

                end_time = datetime.datetime.now()
                st.write(f"⏱️ Total time taken: {end_time - start_time}")
                st.success("✅ Clustering successfully finished! All connected wallets have been scraped.")
                st.session_state.address = btc_address
                #st.write(st.session_state.model_row)
                
        else:
            st.warning("Please provide a Bitcoin address first.")

elif type_of_data_collection == "Data extraction from saved addresses":
    connection.ping(reconnect=True)
    st.info("This feature is under development. Please check back later.")
    sql = "SELECT * FROM saved_data"
    with connection.cursor() as cursor:
        cursor.execute(sql)
        data = cursor.fetchall()
    data = pd.DataFrame(data, columns=["Address", "Save Date", "Number of Transactions"] + [f"Feature {i}" for i in range(24)])
    data_write = data[["Address", "Save Date", "Number of Transactions"]]
    event = st.dataframe(data_write, selection_mode = "single-row-required", on_select = "rerun")
    col1, col2 = st.columns(2)
    btn1 = col1.button("Use Address for prediction")
    if btn1:
        row = data.loc[data.index == event.selection["rows"][0]].iloc[0,3:]
        st.session_state.total_transactions = data.loc[data.index == event.selection["rows"][0]].iloc[0,2]
        st.session_state.row = pd.DataFrame([row.values], columns=range(24))
        alpha = 1
        st.session_state.model_row = (st.session_state.row + alpha) / (st.session_state.total_transactions + 24 * alpha)
        st.success("✅ Address selected for prediction. You can now proceed to Data Visualisation and Model Prediction.")
    btn2 = col2.button("Update Address for prediction") 
    if btn2:
        st.info("Wait for the heuristic clustering to finish.")
        row = data.loc[data.index == event.selection["rows"][0]].iloc[0,3:]
        btc_address = data.loc[data.index == event.selection["rows"][0]].iloc[0,0]
        table_placeholder = st.empty()
        if "write" in st.session_state and st.session_state.write is not None:
            table_placeholder.dataframe(st.session_state.write)
        start_time = datetime.datetime.now()
        with st.spinner("Scraping starting address and connected transaction history..."):
            st.session_state.first_address = pd.DataFrame( 
                columns=["Address", "Number of outgoing txs", "Number of incoming txs", "Address source", "Iteration"]
            )
            table_placeholder.dataframe(st.session_state.first_address)
            st.session_state.write = None

            heur = HeuristicClustering(btc_address, session, table_placeholder)

            heur.heuristic_clus()

            heur.get_final_data()
            st.session_state.model_row = heur.model_df
            st.session_state.row = heur.hour_counts
            st.session_state.total_transactions = heur.total_transactions

            st.session_state.addresses = heur.old_addresses

            end_time = datetime.datetime.now()
            st.write(f"⏱️ Total time taken: {end_time - start_time}")
            st.session_state.address = btc_address
            connection.ping(reconnect=True)
            data_to_save = [st.session_state.address, datetime.datetime.now(), st.session_state.total_transactions] + st.session_state.row.iloc[0].tolist()
            #st.write(data_to_save)
            sql = '''
                INSERT INTO saved_data 
                    (address, save_date, num_of_txs, `0`, `1`, `2`, `3`, `4`, `5`, `6`, `7`, `8`, `9`, `10`, `11`, `12`, `13`, `14`, `15`, `16`, `17`, `18`, `19`, `20`, `21`, `22`, `23`) 
                VALUES 
                    (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE
                    save_date = VALUES(save_date),
                    num_of_txs = VALUES(num_of_txs),
                    `0` = VALUES(`0`), `1` = VALUES(`1`), `2` = VALUES(`2`), 
                    `3` = VALUES(`3`), `4` = VALUES(`4`), `5` = VALUES(`5`), 
                    `6` = VALUES(`6`), `7` = VALUES(`7`), `8` = VALUES(`8`), 
                    `9` = VALUES(`9`), `10` = VALUES(`10`), `11` = VALUES(`11`), 
                    `12` = VALUES(`12`), `13` = VALUES(`13`), `14` = VALUES(`14`), 
                    `15` = VALUES(`15`), `16` = VALUES(`16`), `17` = VALUES(`17`), 
                    `18` = VALUES(`18`), `19` = VALUES(`19`), `20` = VALUES(`20`), 
                    `21` = VALUES(`21`), `22` = VALUES(`22`), `23` = VALUES(`23`)
            '''
            with connection.cursor() as cursor:
                cursor.execute(sql, data_to_save)
            connection.commit()
            st.success("✅ Heuristing clustering finished and data in the database updated!")
            #st.write(st.session_state.model_row)


st.write("---")
st.title("2. Data Visualisation")

if st.button("Generate Transaction Histogram"):
    if st.session_state.model_row is not None:
        fig = create_histogram(st.session_state.row)
        
        # Save to session state so they persist across reruns
        st.session_state.fig = fig
        
        
        st.success("✅ Analysis complete!")
    else:
        st.warning("No data found. Run Data Collection first!")
        
# Always display the histogram if it exists in state
if st.session_state.fig:
    st.pyplot(st.session_state.fig)
    st.dataframe(st.session_state.model_row)
    st.write(f"Model Input Vector Prepared ({st.session_state.total_transactions} transactions). You can save this into database or download it for future use.")
    if type_of_data_collection == "Heuristic clustering":
        save_btn = st.button("Save to Database")
        if save_btn:
            connection.ping(reconnect=True)
            data_to_save = [st.session_state.address, datetime.datetime.now(), st.session_state.total_transactions] + st.session_state.row.iloc[0].tolist()
            #st.write(data_to_save)
            sql = '''
                INSERT IGNORE INTO saved_data 
                (address, save_date, num_of_txs, `0`, `1`, `2`, `3`, `4`, `5`, `6`, `7`, `8`, `9`, `10`, `11`, `12`, `13`, `14`, `15`, `16`, `17`, `18`, `19`, `20`, `21`, `22`, `23`) 
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            '''
            with connection.cursor() as cursor:
                cursor.execute(sql, data_to_save)
            connection.commit()
            st.success("✅ Data saved to database successfully!")


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