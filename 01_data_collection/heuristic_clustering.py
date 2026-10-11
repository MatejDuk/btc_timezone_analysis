from get_address_info import GetAddressInfo
import os
import pymysql
import pandas as pd
import concurrent.futures
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import datetime 

# Top-level configurable parameter for proxy pool length
NUM_PROXIES = 10


class HeuristicClustering:
    """
    Applies standard Bitcoin clustering heuristics (Common Input Ownership & Change Address Heuristics)
    to group addresses owned by the same entity and aggregate transaction entry times for timezone modeling.
    """
    def __init__(self, start_address, session, a, proxies_dict_list, num_proxies=NUM_PROXIES):
        """
        Parameters:
        -----------
        start_address : str
            Seed Bitcoin address to expand through clustering heuristics.
        session : requests.Session / CachedSession
            Cached session object for API requests.
        a : int
            Current proxy list index pointer.
        proxies_dict_list : list of dicts
            List containing proxy configurations formatted for requests.
        num_proxies : int, optional
            Total number of proxies available in the rotation list (default defined by NUM_PROXIES).
        """
        self.a = a
        self.proxies_dict_list = proxies_dict_list
        self.num_proxies = num_proxies if num_proxies else len(proxies_dict_list)
        self.start_address = start_address
        self.session = session
        
        self.model_df = pd.DataFrame(columns=range(24))
        self.db_final = None
        self.total_transactions = None
        self.hour_counts = None
        self.api_key = os.getenv("API_KEY")
        self.change_txids = []

        self.new_iteration = []
        self.new_addresses = []
        self.old_addresses = []

        self.addresses = []
        self.inputs = []
        self.outputs = []
        self.blockchain = []

        self.inputs_final_data = []
        self.outputs_final_data = []
        self.blockchain_final_data = []

        self.write = pd.DataFrame(
            columns=["Address", "Number of outgoing txs", "Number of incoming txs", "Address source", "Iteration"]
        )

    def decim(self, x):
        """
        Calculates decimal precision of floating point BTC values to detect change outputs.
        """
        try:
            s = str(x)
            return len(s.split(".")[1])
        except Exception:
            s = format(x, 'f').rstrip('0').rstrip('.')  
            if '.' in s:
                return len(s.split('.')[1])
            return 0
        
    def html_request(self, address, offset=0, in_out="incoming"):
        """
        Issues an API call to Tatum API to fetch address transaction history for change heuristics.
        """
        url = f"https://api.tatum.io/v3/bitcoin/transaction/address/{address}"
        params = {
            "pageSize": 50,
            "offset": offset,
            "txType": in_out
        }
        headers = {
            "accept": "application/json",
            "x-api-key": self.api_key
        }

        test = True
        j = 0
        while test:
            j += 1
            r = self.session.get(url, timeout=(100, 100), params=params, headers=headers)
            code = r.status_code
            r = r.json()
            if code == 200:
                break
            print(f"Attempt {j} error on {address}")
            time.sleep(5) 
            
        return r
    
    def inputs_analysis(self, inputs, new_iteration):
        """
        Common Input Ownership Heuristic (CIOH):
        Assumes all input addresses in a multi-input transaction belong to the same entity.
        """
        inputs = pd.DataFrame(inputs, columns=["txid", "input_order", "address", "value"])
        input_txids = inputs[inputs["address"].isin(new_iteration)]["txid"].unique()

        input_addresses = inputs[inputs["txid"].isin(input_txids)]["address"].unique().tolist()
        input_addresses = list(set(input_addresses) - set(new_iteration))
        return input_addresses
                
    def change_analysis(self, inputs, outputs, blockchain_data, new_iteration):
        """
        Change Address Identification Heuristic:
        Identifies return change outputs based on address reuse rules, output counts,
        and value decimal precision matching.
        """
        blockchain_data = pd.DataFrame(
            blockchain_data, 
            columns=["txid", "num_inputs", "num_outputs", "fee", "mempool_entry_time", "block_height"]
        )
        input_data = pd.DataFrame(inputs, columns=["txid", "input_order", "address", "value"])
        output_data = pd.DataFrame(outputs, columns=["txid", "output_order", "address", "value"])

        outgoing_txids = input_data[input_data["address"].isin(new_iteration)]["txid"].unique().tolist()
        change_addresses = []

        for txid in outgoing_txids:
            if txid not in self.change_txids:
                self.change_txids.append(txid)
                input_addresses = input_data[input_data["txid"] == txid]["address"].unique().tolist()
                output_addresses = output_data[output_data["txid"] == txid]["address"].unique().tolist()

                # If an output is recycled directly as an input, skip
                if set(output_addresses) & set(input_addresses):
                    continue
                
                # Case 1: Standard 2-output transaction (1 payment + 1 change)
                elif len(output_addresses) == 2:
                    address1, address2 = output_addresses[0], output_addresses[1]

                    r1 = self.html_request(address1, offset=0, in_out="incoming")
                    r2 = self.html_request(address2, offset=0, in_out="incoming")
                    
                    val1 = output_data[(output_data["txid"] == txid) & (output_data["output_order"] == 0)]["value"].iloc[0]
                    val2 = output_data[(output_data["txid"] == txid) & (output_data["output_order"] == 1)]["value"].iloc[0]

                    # Check for fresh address usage (1 incoming tx) and precision disparity
                    if len(r1) == 1 and self.decim(val1) - self.decim(val2) >= 3:
                        change_addresses.append(address1)
                    if len(r2) == 1 and self.decim(val2) - self.decim(val1) >= 3:
                        change_addresses.append(address2)
                        
                # Case 2: Multi-output transaction (> 2 outputs)
                elif len(output_addresses) > 2:
                    count = 0
                    cha = None
                    
                    for address in output_addresses:
                        if address != "":
                            r = self.html_request(address, offset=0, in_out="incoming")
                        
                            if len(r) == 1:
                                cha = address
                                count += 1

                    if count == 1 and cha is not None:
                        change_addresses.append(cha)
                        
        return change_addresses
    
    def heuristic_clus(self):
        """
        Executes iterative breadth-first clustering expansion starting from seed address.
        """
        self.new_iteration = [self.start_address]
        iteration = 0 
        input_addresses = []
        change_addresses = []
        self.old_addresses = [self.start_address]

        while True:
            for address in self.new_iteration:
                # Fetch raw transaction data passing the configurable proxy length
                new_addresses_info = GetAddressInfo(
                    address, 
                    self.session, 
                    self.a, 
                    self.proxies_dict_list, 
                    num_proxies=self.num_proxies
                )
                new_addresses_info.fetch_and_extract()
                
                # Sync current proxy pointer index across iterations
                self.a = new_addresses_info.a
                self.inputs += new_addresses_info.batch_tx_inputs
                self.outputs += new_addresses_info.batch_tx_outputs
                self.blockchain += new_addresses_info.batch_blockchain_data

            # Run heuristics over collected batch
            input_addresses += self.inputs_analysis(self.inputs, self.new_iteration)
            change_addresses += self.change_analysis(self.inputs, self.outputs, self.blockchain, self.new_iteration)

            if iteration == 0:
                new_row = pd.DataFrame({
                    "Address": [address],
                    "Number of outgoing txs": [new_addresses_info.outgoing_count],
                    "Number of incoming txs": [new_addresses_info.incoming_count],
                    "Address source": ["Starting address"],
                    "Iteration": [iteration]
                })
            else:
                new_row = pd.DataFrame({
                    "Address": [address],
                    "Number of outgoing txs": [new_addresses_info.outgoing_count],
                    "Number of incoming txs": [new_addresses_info.incoming_count],
                    "Address source": ["Heuristic clustering"],
                    "Iteration": [iteration]
                })
                
            iteration += 1

            self.new_addresses = input_addresses + change_addresses
            
            # Consolidate cumulative transaction records
            self.blockchain_final_data.extend(self.blockchain)
            self.inputs_final_data.extend(self.inputs)
            self.outputs_final_data.extend(self.outputs)

            self.inputs = []
            self.outputs = []
            self.blockchain = []

            # Compute newly discovered addresses for next expansion tier
            diff_addr = list(set(self.new_addresses) - set(self.old_addresses))
            print(diff_addr)
            if len(diff_addr) == 0:
                break
                
            self.new_iteration = diff_addr
            self.new_addresses = diff_addr
            self.old_addresses += diff_addr
            input_addresses = []
            change_addresses = []
        
        # Build final formatted Pandas DataFrames
        self.blockchain_final = pd.DataFrame(
            self.blockchain_final_data, 
            columns=["txid", "num_inputs", "num_outputs", "fee", "mempool_entry_time", "block_height"]
        )
        
        self.inputs_final = pd.DataFrame(
            self.inputs_final_data, 
            columns=["txid", "input_order", "address", "value"]
        )
        
        self.outputs_final = pd.DataFrame(
            self.outputs_final_data, 
            columns=["txid", "output_order", "address", "value"]
        )

    def get_final_data(self):
        """
        Aggregates transaction mempool entry times by hour of day (0-23) and applies
        Laplace / Bayesian smoothing to produce a normalized 24-dimensional feature vector.
        """
        txs = self.inputs_final[self.inputs_final["address"].isin(self.old_addresses)]["txid"].unique().tolist()
        database = self.blockchain_final[self.blockchain_final["txid"].isin(txs)]
        
        self.db_final = database.drop_duplicates(subset=["txid"])
        self.db_final["hour"] = pd.to_datetime(self.db_final["mempool_entry_time"]).dt.hour
        self.hour_counts = self.db_final.groupby("hour").size()
        
        # Ensure all 24 hours exist in histogram index, filling missing hours with 0
        all_hours = pd.Series(0, index=range(24))
        self.hour_counts = self.hour_counts.add(all_hours, fill_value=0)
        
        self.total_transactions = self.hour_counts.sum()
        
        # Bayesian additive smoothing (alpha = 1) to prevent zero-probability values in downstream models
        alpha = 1
        normalized_vector = (self.hour_counts + alpha) / (self.total_transactions + 24 * alpha)
        
        self.model_df = pd.DataFrame([normalized_vector.values], columns=range(24))
        self.hour_counts = pd.DataFrame([self.hour_counts.values], columns=range(24))