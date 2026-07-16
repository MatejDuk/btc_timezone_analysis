import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class Histogram:
    def __init__(self, connection, cursor):
        self.connection = connection
        self.cursor = cursor

    def create_histogram(self, addresses):
        #Getting all outgoing transaction to addresses from list
        placeholders = ','.join(['%s'] * len(addresses))
        query = f"""SELECT DISTINCT ti.txid, ti.input_order, ti.address, bd.mempool_entry_time, bd.fee, bd.num_inputs, bd.num_outputs, ti.value FROM tx_inputs AS ti
        LEFT JOIN blockchain_data AS bd on ti.txid = bd.txid
        WHERE address in ({placeholders});"""

        self.cursor.execute(query, addresses)
        database = self.cursor.fetchall()
        database = pd.DataFrame(database, columns = ["txid", "input_order", "address", "mempool_entry_time", "fee", "num_inputs", "num_outputs", "value"])

        db_final = database.drop_duplicates(subset = ["txid"])
        db_final["hour"] = pd.to_datetime(db_final["mempool_entry_time"]).dt.hour
        hour_counts = db_final.groupby("hour").size()
        
        # 2. Reindex to ensure ALL 0-23 hours exist, filling missing with 0
        all_hours = pd.Series(0, index=range(24))
        hour_counts = hour_counts.add(all_hours, fill_value=0)
        
        # 3. Create the vector (this is what your model needs)
        total_transactions = hour_counts.sum()
        
        # 4. Bayesian Normalization
        # alpha = 1 ensures no column is ever 0 (helps with log-based models)
        alpha = 1
        normalized_vector = (hour_counts + alpha) / (total_transactions + 24 * alpha)
        
        # Create a DataFrame for your model/display
        model_df = pd.DataFrame([normalized_vector.values], columns=range(24))
        fig, ax = plt.subplots(figsize=(10, 5))
        
        ax.hist(db_final["mempool_entry_time"].dt.hour, bins=np.arange(25), 
                rwidth=0.8, color='skyblue', edgecolor='black')

        ax.set_xticks(np.arange(24) + 0.5)
        ax.set_xticklabels(range(24))
        ax.set_title("Distribution of Transactions by Hour")
        ax.set_xlabel("Hour of Day")
        ax.set_ylabel("Number of transactions")

        return fig, model_df, total_transactions