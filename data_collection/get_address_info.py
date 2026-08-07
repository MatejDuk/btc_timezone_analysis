import os 
import requests
import requests_cache
import time
import datetime
import math

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
            r = self.session.get(url, timeout = (10, 90), params = params, proxies = self.proxies_dict_list[self.a])
            code = r.status_code
            r = r.json()["txs"]
            self.a += 1
            if self.a == 100:
                self.a = 0
            if code == 200:
                break
            print(f"Attempt {j} error on {self.address}:{code}")
            time.sleep(5) 
        #end_time = time.time()
        #print(end_time-start_time)
        print(self.address)
        return r
    
    def fetch_and_extract(self):
        """Fetches the raw API details and extracts datasets into internal batches."""
            
        offset = 0
        test = True
        while test: 
            r = self.html_request(offset)
            if len(r) == 100:
                offset += 100
            else: 
                test = False

            for row in r:
                incoming = False
                outgoing = False
                #Blockchain data
                txid = row["hash"]
                num_inputs = len(row["inputs"])
                num_outputs = len(row["out"])
                fee = row["fee"]/row["size"]
                #This time is only for block acceptance
                mempool_entry_time = datetime.datetime.fromtimestamp(
                    row["time"], 
                    tz=datetime.timezone.utc
                ).strftime("%Y-%m-%d %H:%M:%S")
                block_height = row["block_height"]
                self.batch_blockchain_data.append([txid, num_inputs, num_outputs, fee, mempool_entry_time, block_height])

                #Inputs
                for i,input in enumerate(row["inputs"]):
                    input_order = i
                    try:
                        address = input["prev_out"]["addr"]
                        value = input["prev_out"]["value"] / 100000000
                    except:
                        address = "No address"
                        value = 0
                    self.batch_tx_inputs.append([txid, input_order, address, value])
                    if address == self.address:
                        outgoing = True

                #Outputs
                for i,output in enumerate(row["out"]):
                    output_order = i
                    address = output["addr"]
                    value = output["value"] /  100000000
                    self.batch_tx_outputs.append([txid, output_order, address, value])
                    if address == self.address:
                        incoming = True
                
                if outgoing:
                    self.outgoing_count += 1
                elif incoming:
                    self.incoming_count += 1