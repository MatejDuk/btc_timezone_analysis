import os 
import requests
import requests_cache
import time
import datetime
import math
import pandas as pd

class GetAddressInfo:
    def __init__(self, address, session, a, proxies_dict_list):
        self.proxies_dict_list = proxies_dict_list
        self.a = a
        self.session = session
        self.address = address
        self.api_key = os.getenv("API_KEY")
        
        self.batch_blockchain_data = []
        self.batch_tx_inputs = []
        self.batch_tx_outputs = []

        self.outgoing_count = 0
        self.incoming_count = 0
    
    def html_request(self, offset=0):
        url = f"https://blockchain.info/rawaddr/{self.address}"
        params = {
            "offset": offset,
            }
        
        test = True
        j = 0
        #start_time = time.time()
        while test:
            j += 1
            try:
                r = self.session.get(url, timeout = (20, 90), params = params, proxies = self.proxies_dict_list[self.a])
                code = r.status_code
                
                #print(self.a)
                self.a += 1
                if self.a == 100:
                    self.a = 0
                if code == 200:
                    r = r.json()["txs"]
                    break
                print(f"Attempt {j} error on {self.address}:{code}")
            except Exception as e:
                print(f"Attempt {j} error on {self.address}: {e}")
            time.sleep(5) 
        #end_time = time.time()
        #print(end_time-start_time)
        #print(self.address)
        return r
    
    def fetch_and_extract(self):
        """Fetches the raw API details and extracts datasets into internal batches."""
            
        start_time = time.time()
        offset = 0
        test = True
        while test: 
            r = self.html_request(offset)
            if len(r) == 100:
                offset += 100
            else: 
                test = False
            
            if len(r) == 0:
                break

            #Blockchain data creation
            data = pd.DataFrame(r)

            blockchain_data = pd.DataFrame()
            blockchain_data["txid"] = data["hash"]
            blockchain_data["num_inputs"] = data["vin_sz"]
            blockchain_data["num_outputs"] = data["vout_sz"]
            blockchain_data["fee"] = data["fee"]/data["size"]
            blockchain_data["mempool_entry_time"] = pd.to_datetime(
                    data['time'], 
                    unit='s',       
                    utc=True        
                ).dt.strftime("%Y-%m-%d %H:%M:%S")
            blockchain_data["block_height"] = data["block_height"]
            blockchain_data = blockchain_data.fillna("")

            self.batch_blockchain_data += blockchain_data.values.tolist()

            #Tx inputs table data creation
            try:
                df_inputs = data[['hash', 'inputs']].explode('inputs').dropna(subset=['inputs'])
                df_inputs = df_inputs.reset_index(drop=True)
                inputs_expanded = pd.DataFrame(df_inputs['inputs'].tolist())
                inputs_df = df_inputs[['hash']].join(inputs_expanded).reset_index(drop=True)
                inputs_df["prev_out"] = inputs_df['prev_out'].to_dict()
                df = pd.DataFrame(inputs_df["prev_out"].tolist(), index = inputs_df.index)
                final_inputs = pd.concat([inputs_df, df], axis = 1)

                tx_inputs = pd.DataFrame()
                tx_inputs["txid"] = final_inputs["hash"]
                tx_inputs["input_order"] = final_inputs["index"]
                tx_inputs["address"] = final_inputs["addr"]
                tx_inputs["value"] = final_inputs["value"]/100000000
                tx_inputs = tx_inputs.fillna("")

                self.batch_tx_inputs += tx_inputs.values.tolist()
            except:
                pass

            #Tx outputs table data creation
            df_outputs = data[['hash', 'out']].explode('out').dropna(subset=['out'])
            df_outputs = df_outputs.reset_index(drop=True)
            outputs_expanded = pd.DataFrame(df_outputs['out'].tolist())
            outputs_df = df_outputs[['hash']].join(outputs_expanded).reset_index(drop=True)

            tx_outputs = pd.DataFrame()
            tx_outputs["txid"] = outputs_df["hash"]
            tx_outputs["output_order"] = outputs_df["n"]
            tx_outputs["address"] = outputs_df["addr"]
            tx_outputs["value"] = outputs_df["value"]/100000000
            tx_outputs = tx_outputs.fillna("")
            
            self.batch_tx_outputs += tx_outputs.values.tolist()
        end_time = time.time()
        #print(end_time - start_time)