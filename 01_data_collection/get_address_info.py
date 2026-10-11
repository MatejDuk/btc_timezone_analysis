import os 
import requests
import requests_cache
import time
import datetime
import math
import pandas as pd

# Top-level configurable parameter for proxy pool length
NUM_PROXIES = 10


class GetAddressInfo:
    """
    Fetches raw transaction data for a given Bitcoin address from the Blockchain.info API.
    Parses and expands raw JSON records into flat tables for blockchain metadata,
    transaction inputs, and transaction outputs.
    """
    def __init__(self, address, session, a, proxies_dict_list, num_proxies=NUM_PROXIES):
        """
        Parameters:
        -----------
        address : str
            The target Bitcoin address to investigate.
        session : requests.Session / CachedSession
            Cached or standard HTTP session to reuse TCP connections.
        a : int
            The current active proxy index in the rotation pool.
        proxies_dict_list : list of dicts
            List containing proxy configurations formatted for the requests library.
        num_proxies : int, optional
            Total number of proxies available in the rotation list (default defined by NUM_PROXIES).
        """
        self.proxies_dict_list = proxies_dict_list
        self.num_proxies = num_proxies if num_proxies else len(proxies_dict_list)
        self.a = a
        self.session = session
        self.address = address
        self.api_key = os.getenv("API_KEY")
        
        # Internal batch containers for extracted tables
        self.batch_blockchain_data = []
        self.batch_tx_inputs = []
        self.batch_tx_outputs = []

        self.outgoing_count = 0
        self.incoming_count = 0
    
    def html_request(self, offset=0):
        """
        Executes a paginated HTTP GET request to Blockchain.info with rotating proxy support and exponential retries.
        """
        url = f"https://blockchain.info/rawaddr/{self.address}"
        params = {"offset": offset}
        
        test = True
        j = 0
        while test:
            j += 1
            try:
                # Issue request using current proxy index with timeout bounds (20s connect, 90s read)
                proxy_index = self.a % self.num_proxies
                r = self.session.get(
                    url, 
                    timeout=(20, 90), 
                    params=params, 
                    proxies=self.proxies_dict_list[proxy_index]
                )
                code = r.status_code
                
                # Advance proxy index and wrap around based on configured proxy list length
                self.a = (self.a + 1) % self.num_proxies

                if code == 200:
                    r = r.json()["txs"]
                    break
                print(f"Attempt {j} error on {self.address}:{code}")
            except Exception as e:
                print(f"Attempt {j} error on {self.address}: {e}")
            
            # Pause between retry attempts to handle rate limits or transient errors
            time.sleep(5) 
            
        return r
    
    def fetch_and_extract(self):
        """
        Iterates through all transaction pages for the target address (100 txs per page),
        extracts structural transaction metrics, flattens inputs/outputs, and appends to batch lists.
        """
        start_time = time.time()
        offset = 0
        test = True
        
        while test: 
            r = self.html_request(offset)
            
            # If the response returns full page size (100 txs), advance offset to next page
            if len(r) == 100:
                offset += 100
            else: 
                test = False
            
            if len(r) == 0:
                break

            # --- 1. Blockchain Transaction Metadata Extraction ---
            data = pd.DataFrame(r)

            blockchain_data = pd.DataFrame()
            blockchain_data["txid"] = data["hash"]
            blockchain_data["num_inputs"] = data["vin_sz"]
            blockchain_data["num_outputs"] = data["vout_sz"]
            blockchain_data["fee"] = data["fee"] / data["size"]
            
            # Convert Unix timestamp to UTC formatted ISO string
            blockchain_data["mempool_entry_time"] = pd.to_datetime(
                data['time'], 
                unit='s',       
                utc=True        
            ).dt.strftime("%Y-%m-%d %H:%M:%S")
            
            blockchain_data["block_height"] = data["block_height"]
            blockchain_data = blockchain_data.fillna("")

            self.batch_blockchain_data += blockchain_data.values.tolist()

            # --- 2. Transaction Inputs Extraction ---
            try:
                # Explode nested input lists into flat DataFrame rows
                df_inputs = data[['hash', 'inputs']].explode('inputs').dropna(subset=['inputs'])
                df_inputs = df_inputs.reset_index(drop=True)
                
                inputs_expanded = pd.DataFrame(df_inputs['inputs'].tolist())
                inputs_df = df_inputs[['hash']].join(inputs_expanded).reset_index(drop=True)
                
                inputs_df["prev_out"] = inputs_df['prev_out'].to_dict()
                df = pd.DataFrame(inputs_df["prev_out"].tolist(), index=inputs_df.index)
                final_inputs = pd.concat([inputs_df, df], axis=1)

                tx_inputs = pd.DataFrame()
                tx_inputs["txid"] = final_inputs["hash"]
                tx_inputs["input_order"] = final_inputs["index"]
                tx_inputs["address"] = final_inputs["addr"]
                tx_inputs["value"] = final_inputs["value"] / 100000000  # Convert Satoshis to BTC
                tx_inputs = tx_inputs.fillna("")

                self.batch_tx_inputs += tx_inputs.values.tolist()
            except Exception:
                pass

            # --- 3. Transaction Outputs Extraction ---
            df_outputs = data[['hash', 'out']].explode('out').dropna(subset=['out'])
            df_outputs = df_outputs.reset_index(drop=True)
            
            outputs_expanded = pd.DataFrame(df_outputs['out'].tolist())
            outputs_df = df_outputs[['hash']].join(outputs_expanded).reset_index(drop=True)

            tx_outputs = pd.DataFrame()
            tx_outputs["txid"] = outputs_df["hash"]
            tx_outputs["output_order"] = outputs_df["n"]
            tx_outputs["address"] = outputs_df["addr"]
            tx_outputs["value"] = outputs_df["value"] / 100000000  # Convert Satoshis to BTC
            tx_outputs = tx_outputs.fillna("")
            
            self.batch_tx_outputs += tx_outputs.values.tolist()
            
        end_time = time.time()