# Bitcoin: A Time Zone Analysis

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://timezoneanalysis.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the complete codebase and methodology for the article and Master's thesis, **"Bitcoin: A Time Zone Analysis"**. 

The project investigates the feasibility of deanonymizing the geographic macro-regions of Bitcoin entities by analyzing their transactional circadian rhythms. By leveraging web-scraped ground-truth data, heuristic clustering, and machine learning, the model predicts whether an entity operates within the Americas, Euro-Africa, Central & South Asia, or the East Asia & Pacific regions based strictly on temporal inactivity windows (sleep cycles).

---

## 🗂️ Repository Structure

The repository is chronologically organized into four main stages to reflect the academic pipeline:

### 1. [`/01_data_collection`](./01_data_collection)
Contains the initial data acquisition, entity-building, and deanonymization filtering scripts.
* **Bitcointalk Scraper:** Jupyter notebooks utilizing rotating proxies to systematically scrape Bitcoin addresses and self-reported locations from 3.7 millions of registered forum profiles.
* **Exchange Identification:** Cross-referencing entities against the Arkham Intelligence API to isolate retail participants from commercial exchanges and custodial platforms. *(Note: The Arkham API querying scripts are omitted from this public release to protect private API keys and access credentials).*
* **Blockchain API Integration:** Scripts querying the Blockchain.com Explorer API to retrieve transaction histories, prioritizing mempool entry timestamps.
* **Heuristic Clustering:** Implementation of Common-Input-Ownership ($H_1$) and Change-Address Detection ($H_2$) to aggregate isolated addresses into controlling entities during the collection phase.

### 2. [`/02_data_processing`](./02_data_processing)
Focuses on data hygiene, feature engineering, and visual analysis.
* **Data Filtration:** Applying the thesis-defined minimum and maximum transaction boundaries alongside address-cluster size thresholds.
* **Feature Engineering:** Transforming raw transactional histories into 24-hour relative activity columns using Laplace smoothing, mapping specific time zones to the four consolidated macro-regions, and engineering supplementary behavioral variables.
* **Exploratory Data Analysis:** Notebooks for generating aggregated regional activity histograms, diurnal distributions, and class balance visualizations.

### 3. [`/03_model_training`](./03_model_training)
Contains the machine learning architecture, validation, and robustness checks. *(Note: Large raw training/testing datasets and local MLflow tracking runs are excluded due to GitHub storage limits).*
* **Hyperparameter Tuning:** Parameter optimization for baseline Logistic Regression and the custom-penalized XGBoost classifier, tracked locally via MLflow.
* **Evaluation & Bootstrapping:** Comparing test-set performance across models using a 1,000-iteration paired bootstrap design to derive robust confidence intervals.
* **Sensitivity Analysis:** Evaluating model stability across varying minimum and maximum transaction volume thresholds.

### 4. [`/04_streamlit_app`](./streamlit_app)
The production-ready codebase for the interactive web deployment.
* **`app.py`:** The main Streamlit dashboard allowing users to input hypothetical temporal behavior and visualize the model's regional confidence distribution.
* **TiDB Cloud Connection:** Secure, `pymysql`-based integration to query historical blockchain entity datasets.
* **Model Artifacts:** Contains the serialized `timezone_model.pkl` and deployment dependencies (`requirements.txt`).

---

## 🚀 Live Interactive Demo

An interactive version of the predictive model is deployed via Streamlit Community Cloud. Users can manually adjust transaction frequencies across a 24-hour UTC cycle to observe how temporal shifts impact the XGBoost model's regional predictions.

**👉 Try the app here:** [Bitcoin Time Zone Predictor](https://timezoneanalysis.streamlit.app/)

---

## 💻 Local Setup & Installation

If you wish to run the Streamlit app locally:

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/MatejDuk/btc_timezone_analysis.git](https://github.com/MatejDuk/btc_timezone_analysis.git)
   cd btc_timezone_analysis