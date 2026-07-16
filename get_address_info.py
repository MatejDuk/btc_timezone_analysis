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
                r = self.session.get(url, timeout=(10, 10), params=params, headers = headers)
                r.raise_for_response()
                r = r.json()
            except Exception as e:
                print(f"Attempt {j} error on {self.address}: {e}")
                time.sleep(5) 
            else:
                break
        end_time = time.time()
        #print(end_time-start_time)
        print(self.address)
        return r
    
    def fetch_and_extract(self):
        """Fetches the raw API details and extracts datasets into internal batches."""
            
        offset = 0
        test = True
        while test: 
            r = self.html_request(offset)
            if len(r) == 50:
                offset += 50
            else: 
                test = False

            for row in r:
                incoming = False
                outgoing = False
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
                    try:
                        address = input["coin"]["address"]
                        value = input["coin"]["value"] / 100000000
                    except:
                        address = "No address"
                        value = 0
                    self.batch_tx_inputs.append([txid, input_order, address, value])
                    if address == self.address:
                        outgoing = True

                #Outputs
                for i,output in enumerate(row["outputs"]):
                    output_order = i
                    address = output["address"]
                    value = output["value"] /  100000000
                    self.batch_tx_outputs.append([txid, output_order, address, value])
                    if address == self.address:
                        incoming = True
                
                if outgoing:
                    self.outgoing_count += 1
                if incoming:
                    self.incoming_count += 1