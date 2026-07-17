import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class Histogram:
    def __init__(self, addresses, inputs, outputs, blockchain):
        self.addresses = addresses
        self.inputs = inputs
        self.outputs = outputs
        self.blockchain = blockchain

    def create_histogram(self, addresses):
        txs = self.inputs[self.inputs["address"].isin(self.addresses)]["txid"].unique().tolist()
        database = self.blockchain[self.blockchain["txid"].isin(txs)]
        
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