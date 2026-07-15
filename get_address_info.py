import os 
import requests
import requests_cache
import time
import datetime
import math

class GetAddressInfo:
    def __init__(self, address, session):
        self.session = session
        self.address = address
        self.api_key = os.getenv("API_KEY")
        
        self.batch_blockchain_data = []
        self.batch_tx_inputs = []
        self.batch_tx_outputs = []

        self.outgoing_count = 0
        self.incoming_count = 0
    
    def html_request(self, offset=0):
        url = f"https://api.tatum.io/v3/bitcoin/transaction/address/{self.address}"
        params = {
            "pageSize": 50,
            "offset": offset,
            }
        headers = {
            "accept": "application/json",
            "x-api-key": self.api_key
            }

        test = True
        j = 0
        start_time = time.time()
        while test:
            j += 1
            try:
                r = self.session.get(url, timeout=(10, 90), params=params, headers = headers)
                r = r.json()
                
            except Exception as e:
                print(f"Attempt {j} error on {self.address}: {e}")
                time.sleep(2) 
            else:
                break
        end_time = time.time()
        print(end_time-start_time)
        return r
    
    def fetch_and_extract(self):
        """Fetches the raw API details and extracts datasets into internal batches."""
            
        r = self.html_request(offset=0)
        
        if len(r) < 50:   
            for row in r:
                #Blockchain data
                txid = row["hash"]
                num_inputs = len(row["inputs"])
                num_outputs = len(row["outputs"])
                fee = row["fee"]
                #This time is only for block acceptance
                mempool_entry_time = datetime.datetime.fromtimestamp(
                    row["time"], 
                    tz=datetime.timezone.utc
                ).strftime("%Y-%m-%d %H:%M:%S")
                block_height = row["blockNumber"]
                self.batch_blockchain_data.append([txid, num_inputs, num_outputs, fee, mempool_entry_time, block_height])

                #Inputs
                for i,input in enumerate(row["inputs"]):
                    input_order = i
                    address = input["coin"]["address"]
                    value = input["coin"]["value"] / 100000000
                    self.batch_tx_inputs.append([txid, input_order, address, value])

                #Outputs
                for i,output in enumerate(row["outputs"]):
                    output_order = i
                    address = output["address"]
                    value = output["value"] /  100000000
                    self.batch_tx_outputs.append([txid, output_order, address, value])
            