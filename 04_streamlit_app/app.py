from multiprocessing.dummy import connection
import streamlit as st
import pickle
import pandas as pd
import xgboost as xgb
import numpy as np
import pymysql
import os
import requests_cache
import datetime
import matplotlib.pyplot as plt
import geopandas as gpd
import plotly.express as px
import plotly.graph_objects as go
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

# New session state to track the selected row in the dataframe
if "last_selection" not in st.session_state:
    st.session_state.last_selection = []
    

# Connection to DB
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
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, "timezone_model.pkl")
        with open(model_path, 'rb') as file:
            data = pickle.load(file)
            return data["model"], data["encoder"]
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

model, encoder = load_model()

def create_histogram(model_row):
    values = model_row.iloc[0].tolist()
    hours = list(range(24))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(hours, values, width=1.0, align='edge', color='steelblue', edgecolor='black')
    ax.set_title("Bitcoin Transaction Activity", size=16)
    ax.set_xlabel("Hour of the Day", size=14)
    ax.set_ylabel("Number of transactions", size=14)
    ax.set_xticks(np.arange(24) + 0.5)
    ax.set_xticklabels(range(24))
    return fig

# Load and prepare geodata for mapping
@st.cache_data
def load_map_data():
    url = "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_10m_time_zones.geojson"
    tz_gdf = gpd.read_file(url)

    tz_gdf["utc_format"] = tz_gdf["utc_format"].fillna("N/A")
    tz_gdf["places"] = tz_gdf["places"].fillna("N/A")

    region_mapping = {
        -9.0: 'America', -8.0: 'America', -7.0: 'America', -6.0: 'America', -5.0: 'America', -4.0: 'America', -3.0: 'America', -2.0: 'America', -4.5: 'America', -9.5: 'America',
        0.0: 'Euro_Africa', 1.0: 'Euro_Africa', 2.0: 'Euro_Africa', 3.0: 'Euro_Africa',  -1.0: 'Euro_Africa',
        3.5: 'Central_South_Asia', 4.0: 'Central_South_Asia', 4.5: 'Central_South_Asia', 5.0: 'Central_South_Asia',
        5.5: 'Central_South_Asia', 5.75: 'Central_South_Asia', 6.0: 'Central_South_Asia', 6.5: 'Central_South_Asia',
        7.0: 'East_Asia_Pac', 8.0: 'East_Asia_Pac', 9.0: 'East_Asia_Pac', 10.0: 'East_Asia_Pac', 10.5: 'East_Asia_Pac', 11.0: 'East_Asia_Pac',
        12.0: 'East_Asia_Pac', 13.0: 'East_Asia_Pac', -10.0: 'East_Asia_Pac', -11.0: 'East_Asia_Pac', -12.0: 'East_Asia_Pac', 9.5: 'East_Asia_Pac', 14.0: 'East_Asia_Pac'
    }

    tz_gdf['Region'] = tz_gdf['zone'].map(region_mapping).fillna('Unassigned')
    return tz_gdf

def reset_state():
    st.session_state.write = None
    st.session_state.btc_address_input = ""
    st.session_state.addresses = []
    st.session_state.total_transactions = 0
    st.session_state.row = None
    st.session_state.fig = None
    st.session_state.model_row = None
    st.session_state.last_selection = []

def reset_write_state():
    st.session_state.write = None

# Header layout
st.title("🌍 Bitcoin Time Zone Predictor")
st.markdown("""
Welcome to the **Bitcoin Time Zone Predictor**. Developed as part of my Master's thesis, this application utilizes a machine learning model to classify a Bitcoin entity's geographic region—**America, Euro_Africa, Central_South_Asia or East_Asia_Pac**—based entirely on its transaction history. 

### How to use this app:
The analysis pipeline is divided into three simple stages:

* **1. Data Collection:** Choose whether to analyze a brand-new Bitcoin address using *Heuristic Clustering* to scrape the blockchain, or load a previously analyzed address from the *Saved Addresses* database.
* **2. Data Visualisation:** Generate a chronological histogram of the combined transaction activity. This step standardizes the scraped data and prepares the exact 24-hour input vector required by the ML model.
* **3. Model Prediction:** Run the pre-trained XGBoost classification model to calculate the probability distribution across the four global macro-regions and visualize the prediction on an interactive map.

---
*Disclaimer: Be aware that this is a hobby project and relies on free-tier APIs, which may occasionally result in rate limits or timeouts. If you encounter any bugs or wish to explore the underlying architecture, the complete repository is available on my GitHub. For direct inquiries, feel free to contact me at [matej.dukat@gmail.com](mailto:matej.dukat@gmail.com).*
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
                
        else:
            st.warning("Please provide a Bitcoin address first.")

elif type_of_data_collection == "Data extraction from saved addresses":
    st.info("""
    **Saved Addresses Database:** Here you can explore Bitcoin addresses that have already been processed and saved by users.
    
    To begin, **click on a row** in the table below to select an address. Then choose an action:
    * **Use Address for prediction:** Instantly loads the 24-hour transaction data for the selected address. You can proceed directly to *Step 2 (Visualisation)* and *Step 3 (Prediction)* without waiting for the blockchain to be scraped again.
    * **Update Address for prediction:** Re-runs the full heuristic clustering to fetch the most up-to-date transaction history for the selected address and updates the database record. *(Note: This process is computationally intensive and takes time).*
    """)
    
    connection.ping(reconnect=True)
    sql = "SELECT * FROM saved_data"
    with connection.cursor() as cursor:
        cursor.execute(sql)
        data = cursor.fetchall()
        
    data = pd.DataFrame(data, columns=["Address", "Save Date", "Number of Transactions"] + [f"Feature {i}" for i in range(24)])
    data_write = data[["Address", "Save Date", "Number of Transactions"]]
    
    event = st.dataframe(data_write, selection_mode = "single-row-required", on_select = "rerun")
    
    # Check if the user selected a different row than before
    current_selection = event.selection["rows"]
    if current_selection != st.session_state.last_selection:
        # Reset visualisations and model status if the selection changed
        st.session_state.fig = None
        st.session_state.model_row = None
        st.session_state.last_selection = current_selection

    col1, col2 = st.columns(2)
    btn1 = col1.button("Use Address for prediction")
    
    if btn1 and len(event.selection["rows"]) > 0:
        row = data.loc[data.index == event.selection["rows"][0]].iloc[0,3:]
        st.session_state.total_transactions = data.loc[data.index == event.selection["rows"][0]].iloc[0,2]
        st.session_state.row = pd.DataFrame([row.values], columns=range(24))
        alpha = 1
        st.session_state.model_row = (st.session_state.row + alpha) / (st.session_state.total_transactions + 24 * alpha)
        st.session_state.fig = None # Ensure a fresh visualization is required
        st.success("✅ Address selected for prediction. You can now proceed to Data Visualisation and Model Prediction.")
        
    btn2 = col2.button("Update Address for prediction") 
    if btn2 and len(event.selection["rows"]) > 0:
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
            st.success("✅ Heuristic clustering finished and data in the database updated!")


st.write("---")
st.title("2. Data Visualisation")

# Change the button logic to flip a boolean state
if st.button("Generate Transaction Histogram"):
    if st.session_state.model_row is not None:
        st.session_state.show_histogram = True  # Use a boolean flag instead of storing the fig
        st.success("✅ Analysis complete!")
    else:
        st.warning("No data found. Run Data Collection first!")
        
if st.session_state.get("show_histogram", False):
    # Recreate the figure freshly on every rerun
    fig = create_histogram(st.session_state.row)
    st.pyplot(fig)
    st.dataframe(st.session_state.model_row)
    st.write(f"Model Input Vector Prepared ({st.session_state.total_transactions} transactions)...")
    if type_of_data_collection == "Heuristic clustering":
        save_btn = st.button("Save to Database")
        if save_btn:
            connection.ping(reconnect=True)
            data_to_save = [st.session_state.address, datetime.datetime.now(), st.session_state.total_transactions] + st.session_state.row.iloc[0].tolist()
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

    # Map Generation using dynamic probabilities
    st.subheader("Geospatial Probability Map")
    with st.spinner("Rendering interactive map..."):
        tz_gdf = load_map_data().copy()
        
        # Map the model's confidence scores directly to regions
        prob_dict = dict(zip(prob_df['Time Zone'], prob_df['Confidence']))
        prob_dict['Unassigned'] = 0.0
        
        # Map the internal region labels to your exact requested descriptions
        clean_names = {
            'America': 'America',
            'Euro_Africa': 'Euro_Africa',
            'Central_South_Asia': 'Central_South_Asia',
            'East_Asia_Pac': 'East_Asia_Pac',
            'Unassigned': 'Unassigned'
        }
        
        tz_gdf['Probability'] = tz_gdf['Region'].map(prob_dict).fillna(0.0)
        tz_gdf['Prob_Str'] = (tz_gdf['Probability'] * 100).round(1).astype(str) + '%'
        tz_gdf['Display_Region'] = tz_gdf['Region'].map(clean_names)

        fig = px.choropleth(
            tz_gdf,
            geojson=tz_gdf.geometry,
            locations=tz_gdf.index,
            color="Region",
            color_discrete_map={
                "America": "#EF553B",
                "Euro_Africa": "#00CC96",
                "Central_South_Asia": "#AB63FA",
                "East_Asia_Pac": "#19D3F3",
                "Unassigned": "rgba(200, 200, 200, 0.15)",
            },
            hover_name="name",
            hover_data={
                "Region": False,  
                "Display_Region": True, 
                "Prob_Str": True,
                "utc_format": True,
                "places": True,
                "zone": False, 
            },
            projection="natural earth",
        )

        def get_prob_str(region_name):
            return f"{(prob_dict.get(region_name, 0) * 100):.1f}%"

        # Formatted label text for the map overlay using your exact descriptions
        text_lons = [-90, 15, 75, 135]
        text_lats = [15, 15, 25, -5]
        text_labels = [
            f"<b>America</b><br>{get_prob_str('America')}", 
            f"<b>Euro-Africa</b><br>{get_prob_str('Euro_Africa')}", 
            f"<b>Central Asia / India /<br>Middle East</b><br>{get_prob_str('Central_South_Asia')}", 
            f"<b>East Asia / Pacific</b><br>{get_prob_str('East_Asia_Pac')}"
        ]

        fig.add_trace(go.Scattergeo(
            lon=text_lons,
            lat=text_lats,
            text=text_labels,
            mode="text",
            textfont=dict(size=13, color="black"),
            showlegend=False,
            hoverinfo="skip" 
        ))

        fig.update_geos(
            showcountries=True,
            countrycolor="#222222",
            countrywidth=0.7,
            showcoastlines=True,
            coastlinecolor="#333333",
            coastlinewidth=0.8,
            showland=True,
            landcolor="#f8f9fa",
            fitbounds="locations",
            visible=True,
            projection_scale=1, 
        )

        fig.update_traces(
            marker_line_width=0.4,
            marker_line_color="rgba(100, 100, 100, 0.4)",
            marker_opacity=0.45,
            selector=dict(type='choropleth') 
        )

        fig.data[0].hovertemplate = (
            "<b>%{hovertext}</b><br><br>"
            "Region: %{customdata[0]}<br>"
            "Probability: %{customdata[1]}<br>"
            "Offset: %{customdata[2]}<br>"
            "Key Places: %{customdata[3]}<extra></extra>"
        )

        fig.update_layout(
            margin={"r": 0, "t": 20, "l": 0, "b": 0},
            showlegend=False, 
            dragmode=False, 
        )
        
        st.plotly_chart(fig, config={
            "scrollZoom": False,
            "displayModeBar": False,  
            "doubleClick": False,
        }, use_container_width=True)