import os
import pickle
import datetime
import pymysql
import requests_cache
import numpy as np
import pandas as pd
import geopandas as gpd
import xgboost as xgb
import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from get_address_info import GetAddressInfo
from heuristic_clustering import HeuristicClustering

# 1. Page configuration MUST be the first Streamlit command
st.set_page_config(
    page_title="BTC Time Zone Predictor",
    page_icon="🌍",
    layout="centered"
)

# 2. Initialize session states consistently
DEFAULT_STATES = {
    "address": None,
    "addresses": [],
    "total_transactions": 0,
    "write": None,
    "row": None,
    "show_histogram": False,
    "model_row": None,
    "last_selection": [],
}

for key, default_val in DEFAULT_STATES.items():
    if key not in st.session_state:
        st.session_state[key] = default_val


# Connection to DB
@st.cache_resource
def get_db_connection():
    return pymysql.connect(
        host="gateway01.eu-central-1.prod.aws.tidbcloud.com",
        port=4000,
        user=os.getenv("USER_DATABASE"),
        password=os.getenv("PASSWORD_DATABASE"),
        database="test",
        ssl_verify_cert=True,
        ssl_verify_identity=True
    )

connection = get_db_connection()


@st.cache_data(ttl=120)
def fetch_saved_addresses():
    connection.ping(reconnect=True)
    sql = "SELECT * FROM saved_data"
    with connection.cursor() as cursor:
        cursor.execute(sql)
        rows = cursor.fetchall()
    cols = ["Address", "Save Date", "Number of Transactions"] + [f"Feature {i}" for i in range(24)]
    return pd.DataFrame(rows, columns=cols)


# Cache session for saving API requests (expires after 30 days)
@st.cache_resource
def get_cached_session():
    return requests_cache.CachedSession('api_cache', expire_after=datetime.timedelta(days=30))

session = get_cached_session()


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


def create_histogram(row_data):
    # Safely convert any input (DataFrame, Series, list, or Decimal/object) into 24 floats
    values = pd.to_numeric(np.asarray(row_data).flatten(), errors="coerce")
    values = np.nan_to_num(values, nan=0.0)[:24]
    hours = list(range(24))

    fig = go.Figure(
        data=[
            go.Bar(
                x=hours,
                y=values,
                marker=dict(
                    color="steelblue",
                    line=dict(color="black", width=1)
                ),
                width=1.0,
                hovertemplate="<b>Hour %{x}:00 - %{x}:59 UTC</b><br>Transactions: %{y}<extra></extra>"
            )
        ]
    )

    fig.update_layout(
        title=dict(text="Bitcoin Transaction Activity", font=dict(size=18)),
        xaxis=dict(
            title="Hour of the Day (UTC)",
            tickmode="array",
            tickvals=hours,
            ticktext=[str(h) for h in hours],
            range=[-0.5, 23.5]
        ),
        yaxis=dict(title="Number of Transactions"),
        bargap=0,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=50, b=20),
        height=400
    )
    return fig


# Load and prepare simplified geodata for low-memory mapping
@st.cache_data
def load_map_data():
    url = "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_10m_time_zones.geojson"
    tz_gdf = gpd.read_file(url)

    # Keep only the columns we actually need to reduce memory usage
    tz_gdf = tz_gdf[["name", "utc_format", "places", "zone", "geometry"]].copy()
    tz_gdf["utc_format"] = tz_gdf["utc_format"].fillna("N/A")
    tz_gdf["places"] = tz_gdf["places"].fillna("N/A")

    # Lower map resolution: reduces geometry vertex count by ~95%
    tz_gdf["geometry"] = tz_gdf["geometry"].simplify(tolerance=0.15, preserve_topology=True)

    region_mapping = {
        -8.0: 'America', -7.0: 'America', -6.0: 'America', -5.0: 'America',
        -4.0: 'America', -3.0: 'America', -2.0: 'America', -4.5: 'America', -3.5: 'America', 
        0.0: 'Euro_Africa', 1.0: 'Euro_Africa', 2.0: 'Euro_Africa', 3.0: 'Euro_Africa', -1.0: 'Euro_Africa',
        3.5: 'Central_South_Asia', 4.0: 'Central_South_Asia', 4.5: 'Central_South_Asia', 5.0: 'Central_South_Asia',
        5.5: 'Central_South_Asia', 5.75: 'Central_South_Asia', 6.0: 'Central_South_Asia', 6.5: 'Central_South_Asia',
        7.0: 'East_Asia_Pac', 8.0: 'East_Asia_Pac', 9.0: 'East_Asia_Pac', 10.0: 'East_Asia_Pac',
        10.5: 'East_Asia_Pac', 11.0: 'East_Asia_Pac', 12.0: 'East_Asia_Pac', 13.0: 'East_Asia_Pac',
        -10.0: 'East_Asia_Pac', -11.0: 'East_Asia_Pac', -12.0: 'East_Asia_Pac', 9.5: 'East_Asia_Pac', 14.0: 'East_Asia_Pac', 
        -9.0: 'East_Asia_Pac', -9.5: 'East_Asia_Pac'
    }

    tz_gdf['Region'] = tz_gdf['zone'].map(region_mapping).fillna('Unassigned')
    return tz_gdf


def reset_state():
    st.session_state.write = None
    st.session_state.btc_address_input = ""
    st.session_state.addresses = []
    st.session_state.total_transactions = 0
    st.session_state.row = None
    st.session_state.show_histogram = False
    st.session_state.model_row = None
    st.session_state.last_selection = []


def reset_write_state():
    st.session_state.write = None
    st.session_state.show_histogram = False


# Header layout
st.title("🌍 Bitcoin Time Zone Predictor")
st.markdown("""
Welcome to the **Bitcoin Time Zone Predictor**. Developed as part of my Master's thesis, this application utilizes a machine learning model to classify a Bitcoin entity's geographic region—**America, Euro_Africa, Central_South_Asia or East_Asia_Pac**—based entirely on its transaction history. 

### How to use this app:
The analysis pipeline is divided into three simple stages:

- **1. Data Collection:** Choose whether to analyze a brand-new Bitcoin address using *Heuristic Clustering* to scrape the blockchain, or load a previously analyzed address from the *Saved Addresses* database.
- **2. Data Visualisation:** Generate a chronological histogram of the combined transaction activity. This step standardizes the scraped data and prepares the exact 24-hour input vector required by the ML model.
- **3. Model Prediction:** Run the pre-trained XGBoost classification model to calculate the probability distribution across the four global macro-regions and visualize the prediction on an interactive map.

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
- 18LHS5Guof1GHQgcXHQkHpKm4VV45Utnki
- 1N9vbvE2Yge7W2RVfDfvRuDVJXGQjPxz4D
- 1EmcSm2yyREEgkheJ2YUSyhaxkKno47NnK
- 15Yhg5Mj7tkheFi4yybpNFC3U4z6sQwRVb
    """)
    btc_address = st.text_input(
        label="Enter existing btc address",
        key="btc_address_input",
        on_change=reset_write_state
    )
    heur_btn = st.button("Start Heuristic Clustering")
    table_placeholder = st.empty()
    if st.session_state.write is not None:
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

                # Ensure 2D DataFrame with numeric values
                raw_counts = pd.to_numeric(np.asarray(heur.hour_counts).flatten(), errors="coerce")
                st.session_state.row = pd.DataFrame([raw_counts], columns=range(24))
                st.session_state.model_row = heur.model_df
                st.session_state.total_transactions = int(heur.total_transactions)
                st.session_state.addresses = heur.old_addresses
                st.session_state.show_histogram = False

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
- **Use Address for prediction:** Instantly loads the 24-hour transaction data for the selected address. You can proceed directly to *Step 2 (Visualisation)* and *Step 3 (Prediction)* without waiting for the blockchain to be scraped again.
- **Update Address for prediction:** Re-runs the full heuristic clustering to fetch the most up-to-date transaction history for the selected address and updates the database record. *(Note: This process is computationally intensive and takes time).*
    """)

    data = fetch_saved_addresses()
    data_write = data[["Address", "Save Date", "Number of Transactions"]]

    event = st.dataframe(data_write, selection_mode="single-row-required", on_select="rerun")

    # Reset visualisations and model status if the selection changed
    current_selection = event.selection["rows"]
    if current_selection != st.session_state.last_selection:
        st.session_state.show_histogram = False
        st.session_state.row = None
        st.session_state.model_row = None
        st.session_state.last_selection = current_selection

    col1, col2 = st.columns(2)
    btn1 = col1.button("Use Address for prediction")

    if btn1 and len(event.selection["rows"]) > 0:
        selected_idx = event.selection["rows"][0]
        feature_cols = [f"Feature {i}" for i in range(24)]

        # Explicitly cast to numeric so Matplotlib & XGBoost don't receive object dtype
        row_values = pd.to_numeric(data.loc[selected_idx, feature_cols], errors="coerce").values
        total_txs = float(data.loc[selected_idx, "Number of Transactions"])

        st.session_state.total_transactions = int(total_txs)
        st.session_state.row = pd.DataFrame([row_values], columns=range(24))

        alpha = 1
        st.session_state.model_row = (st.session_state.row + alpha) / (total_txs + 24 * alpha)
        st.session_state.show_histogram = False
        st.success("✅ Address selected for prediction. You can now proceed to Data Visualisation and Model Prediction.")

    btn2 = col2.button("Update Address for prediction")
    if btn2 and len(event.selection["rows"]) > 0:
        st.info("Wait for the heuristic clustering to finish.")
        selected_idx = event.selection["rows"][0]
        btc_address = data.loc[selected_idx, "Address"]
        table_placeholder = st.empty()

        if st.session_state.write is not None:
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

            raw_counts = pd.to_numeric(np.asarray(heur.hour_counts).flatten(), errors="coerce")
            st.session_state.row = pd.DataFrame([raw_counts], columns=range(24))
            st.session_state.model_row = heur.model_df
            st.session_state.total_transactions = int(heur.total_transactions)
            st.session_state.addresses = heur.old_addresses
            st.session_state.show_histogram = False

            end_time = datetime.datetime.now()
            st.write(f"⏱️ Total time taken: {end_time - start_time}")
            st.session_state.address = btc_address

            connection.ping(reconnect=True)
            data_to_save = (
                [st.session_state.address, datetime.datetime.now(), st.session_state.total_transactions]
                + st.session_state.row.iloc[0].tolist()
            )
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
            fetch_saved_addresses.clear()  # Refresh cached table
            st.success("✅ Heuristic clustering finished and data in the database updated!")


st.write("---")
st.title("2. Data Visualisation")

if st.button("Generate Transaction Histogram"):
    if st.session_state.model_row is not None and st.session_state.row is not None:
        st.session_state.show_histogram = True
        st.success("✅ Analysis complete!")
    else:
        st.warning("No data found. Run Data Collection first!")

if st.session_state.get("show_histogram", False) and st.session_state.row is not None:
    fig_hist = create_histogram(st.session_state.row)
    st.plotly_chart(fig_hist, use_container_width=True)

    # Ensure column names are strings so Streamlit Cloud's Arrow serializer doesn't fail
    display_df = st.session_state.model_row.copy()
    display_df.columns = [str(col) for col in display_df.columns]
    st.dataframe(display_df)

    st.write(f"Model Input Vector Prepared ({st.session_state.total_transactions} transactions)...")

    if type_of_data_collection == "Heuristic clustering":
        if st.button("Save to Database"):
            connection.ping(reconnect=True)
            row_list = pd.to_numeric(np.asarray(st.session_state.row).flatten(), errors="coerce").tolist()
            data_to_save = [
                st.session_state.address,
                datetime.datetime.now(),
                int(st.session_state.total_transactions)
            ] + row_list

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

if st.button("Calculate Probabilities", disabled=st.session_state.model_row is None):
    with st.spinner("Running machine learning model..."):
        input_vector = pd.to_numeric(np.asarray(st.session_state.model_row).flatten(), errors="coerce").tolist()
        probs = model.predict_proba([input_vector])[0]

        if encoder:
            class_names = encoder.classes_
        else:
            class_names = [f"Class {i}" for i in range(len(probs))]

        prob_df = pd.DataFrame({
            "Time Zone": class_names,
            "Confidence": probs
        }).sort_values(by="Confidence", ascending=False)

    st.subheader("Predicted Time Zone Probabilities")
    st.bar_chart(prob_df, x="Time Zone", y="Confidence", horizontal=True)
    #st.dataframe(prob_df.set_index("Time Zone"))

    st.subheader("Geospatial Probability Map")
    with st.spinner("Rendering interactive map..."):
        tz_gdf = load_map_data().copy()

        prob_dict = dict(zip(prob_df['Time Zone'], prob_df['Confidence']))
        prob_dict['Unassigned'] = 0.0

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
            resolution=110,
            bgcolor="rgba(0,0,0,0)",  # Removes white background behind the globe
            showframe=False,          # Optional: removes the thin border around the projection
            showcountries=True,
            countrycolor="#222222",
            countrywidth=0.7,
            showcoastlines=True,
            coastlinecolor="#333333",
            coastlinewidth=0.8,
            showland=True,
            landcolor="rgba(0,0,0,0)", # Makes landmasses transparent (was "#f8f9fa" off-white)
            fitbounds="locations",
            visible=True,
            projection_scale=1,
        )
        fig.update_traces(
            marker_line_width=0.4,
            marker_line_color="rgba(100, 100, 100, 0.4)",
            marker_opacity=0.45,
            hovertemplate=(
                "<b>%{hovertext}</b><br><br>"
                "Region: %{customdata[1]}<br>"
                "Probability: %{customdata[2]}<br>"
                "Offset: %{customdata[3]}<br>"
                "Key Places: %{customdata[4]}<extra></extra>"
            ),
            selector=dict(type='choropleth')
        )

        fig.update_layout(
            margin={"r": 0, "t": 20, "l": 0, "b": 0},
            paper_bgcolor="rgba(0,0,0,0)",  # Removes outer container background
            plot_bgcolor="rgba(0,0,0,0)",   # Removes inner plot background
            showlegend=False,
            dragmode=False,
        )

        st.plotly_chart(fig, config={
            "scrollZoom": False,
            "displayModeBar": False,
            "doubleClick": False,
        }, use_container_width=True)