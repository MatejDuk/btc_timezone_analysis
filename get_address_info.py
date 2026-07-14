import os 
import requests
import requests_cache
import time
import datetime
import math

class GetAddressInfo:
    def __init__(self, address, a, proxies_list, session):
        self.session = session
        self.a = a
        self.address = address
        self.proxies_dict_list = proxies_list
        self.length_of_ip_a = len(proxies_list)
        
        self.batch_blockchain_data = []
        self.batch_tx_inputs = []
        self.batch_tx_outputs = []

        self.outgoing_count = 0
        self.incoming_count = 0
    
    def html_request(self, offset=0):
        url = f"https://blockchain.info/rawaddr/{self.address}"
        params = {"offset": offset}
        test = True
        j = 0
        
        while test:
            j += 1
            try:
                # Safely fallback to unproxied requests if no proxies exist
                proxy = self.proxies_dict_list[self.a] if self.length_of_ip_a > 0 else None
                r = self.session.get(url, proxies=proxy, timeout=(10, 90), params=params)
                r = r.json()
                
                if "n_tx" not in r:
                    raise ValueError("n_tx missing from response")
                    
                test = False
                
            except Exception as e:
                print(f"Attempt {j} error on {self.address}: {e}")
                time.sleep(2)  # Reduced delay for parallel threading speed
                
                if self.length_of_ip_a > 0:
                    self.a = (self.a + 1) % self.length_of_ip_a
                else:
                    test = False  # Break loop if we don't have proxies to rotate
                    
        return r
    
    def blockchain_data_data(self, r, l):
        txid = r["txs"][l]["hash"]
        num_inputs = r["txs"][l]["vin_sz"]
        num_outputs = r["txs"][l]["vout_sz"]
        fee = r["txs"][l]["fee"] / r["txs"][l]["size"]
        
        mempool_entry_time = r["txs"][l]["time"]
        mempool_entry_time = datetime.datetime.fromtimestamp(
            mempool_entry_time, 
            tz=datetime.timezone.utc
        ).strftime("%Y-%m-%d %H:%M:%S")
        
        block_height = r["txs"][l]["block_height"]
        return [txid, num_inputs, num_outputs, fee, mempool_entry_time, block_height]
    
    def tx_inputs_data(self, r, l):
        batch = []
        txid = r["txs"][l]["hash"]
        num_inputs = r["txs"][l]["vin_sz"]
        is_incoming = False

        for j in range(num_inputs):
            try:
                input_order = j
                address = r["txs"][l]["inputs"][j]["prev_out"]["addr"]
                value = r["txs"][l]["inputs"][j]["prev_out"]["value"] / 100000000 
                batch.append([txid, input_order, address, value])

                if address == self.address:
                    is_incoming = True
            except KeyError:
                pass
                
        if is_incoming:
            self.incoming_count += 1

        return batch
    
    def tx_outputs_data(self, r, l):
        batch = []
        txid = r["txs"][l]["hash"]
        num_outputs = r["txs"][l]["vout_sz"]
        is_outgoing = False
        
        for k in range(num_outputs):
            try:
                output_order = k
                out = r["txs"][l]["out"][k]
                address = out.get("addr", "UNKNOWN")
                value = out["value"] / 100000000
                batch.append([txid, output_order, address, value])

                if address == self.address:
                    is_outgoing = True
            except KeyError:
                pass
        
        if is_outgoing:
            self.outgoing_count += 1

        return batch
    
    def fetch_and_extract(self):
        """Fetches the raw API details and extracts datasets into internal batches."""
        if self.length_of_ip_a > 0 and self.a >= self.length_of_ip_a:
            self.a = 0
            
        r = self.html_request(offset=0)
        
        if r["n_tx"] > 0:
            iteracie = min(r["n_tx"], 100)
                
            for l in range(iteracie):
                self.batch_blockchain_data.append(self.blockchain_data_data(r, l))
                self.batch_tx_inputs += self.tx_inputs_data(r, l)
                self.batch_tx_outputs += self.tx_outputs_data(r, l)
            
            if r["n_tx"] > 100:
                dodatocne_iteracie = math.floor(r["n_tx"] / 100)
                offset = 100
                for m in range(dodatocne_iteracie):
                    iteracie = r["n_tx"] % 100 if (m + 1) == dodatocne_iteracie else 100

                    if self.length_of_ip_a > 0:
                        self.a = (self.a + 1) % self.length_of_ip_a
                        
                    r = self.html_request(offset=offset)
                            
                    for l in range(iteracie):
                        self.batch_blockchain_data.append(self.blockchain_data_data(r, l))
                        self.batch_tx_inputs += self.tx_inputs_data(r, l)
                        self.batch_tx_outputs += self.tx_outputs_data(r, l)
                        
                    offset += 100