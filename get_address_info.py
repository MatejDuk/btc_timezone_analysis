import os 
import pymysql
import requests
import requests_cache
import time
import datetime
import math

class GetAddressInfo:
    def __init__(self, address, a, connection, cursor, session):
        #Cache session for saving API requests, reset once a month
        self.session = session
        self.a = a
        self.address = address
        self.connection = connection
        self.cursor = cursor
        
        #Getting list of IP addresses to faster get API requests from blockchain explorer
        raw_ips = os.getenv("IP_ADDRESSES")
        if raw_ips:
            self.IP_ADDR = [ip.strip() for ip in raw_ips.split(",")]
        else:
            self.IP_ADDR = []

        self.proxies_dict_list = []


        for line in self.IP_ADDR:
            # split the TXT format IP:PORT:USER:PASS
            ip, port, user, pwd = line.split(":")

            # create a dict directly for requests
            proxy_dict = {
                "http": f"http://{user}:{pwd}@{ip}:{port}",
                "https": f"http://{user}:{pwd}@{ip}:{port}"
            }

            self.proxies_dict_list.append(proxy_dict)
            
        self.length_of_ip_a = len(self.IP_ADDR)
        
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
                r = self.session.get(url, proxies=self.proxies_dict_list[self.a], timeout=(10, 90), params=params)
                r = r.json()
                
                # Verify 'n_tx' exists in the response
                if "n_tx" not in r:
                    raise ValueError("n_tx missing from response")
                    
                test = False
                
            except Exception as e:
                print(f"Attempt {j} error on {self.address}: {e}")
                time.sleep(5)
                self.a += 1

                if self.a >= self.length_of_ip_a:
                    self.a = 0
                    
        return r
    
    def blockchain_data_data(self, r, l):
        txid = r["txs"][l]["hash"]
        num_inputs = r["txs"][l]["vin_sz"]
        num_outputs = r["txs"][l]["vout_sz"]
        fee = r["txs"][l]["fee"] / r["txs"][l]["size"]
        
        # Ensures UTC enforcement is maintained
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
                print("KeyError in inputs extraction")
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
                address = out.get("addr", "UNKNOWN")  # Added fallback for missing output addresses
                value = out["value"] / 100000000
                batch.append([txid, output_order, address, value])

                if address == self.address:
                    is_outgoing = True
            except KeyError:
                print("KeyError in outputs extraction")
        
        if is_outgoing:
            self.outgoing_count += 1

        return batch
    
    def address_write(self):
        if self.a >= self.length_of_ip_a:
            self.a = 0
            
        r = self.html_request(offset=0)
        
        if r["n_tx"] > 0:
            # 5. Fixed the UTC enforcement here as well
            ts = r["txs"][0]["time"]
            ts = datetime.datetime.fromtimestamp(ts, tz=datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
            
            iteracie = min(r["n_tx"], 100)
                
            for l in range(iteracie):
                self.batch_blockchain_data.append(self.blockchain_data_data(r, l))
                self.batch_tx_inputs += self.tx_inputs_data(r, l)
                self.batch_tx_outputs += self.tx_outputs_data(r, l)
            
            if r["n_tx"] > 100:
                dodatocne_iteracie = math.floor(r["n_tx"] / 100)
                if r["n_tx"] > 10000:
                    # Fixed nested quotes in f-string
                    print(f"Address {self.address} with {r['n_tx']} transactions")
                
                offset = 100
                for m in range(dodatocne_iteracie):
                    iteracie = r["n_tx"] % 100 if (m + 1) == dodatocne_iteracie else 100

                    self.a += 1
                    if self.length_of_ip_a > 0 and self.a >= self.length_of_ip_a:
                        self.a = 0
                        
                    r = self.html_request(offset=offset)
                            
                    for l in range(iteracie):
                        self.batch_blockchain_data.append(self.blockchain_data_data(r, l))
                        self.batch_tx_inputs += self.tx_inputs_data(r, l)
                        self.batch_tx_outputs += self.tx_outputs_data(r, l)
                        
                    offset += 100
                    
            chunk_size = 1000

                
            while self.batch_blockchain_data:
                chunk = self.batch_blockchain_data[:chunk_size]
                sql = "INSERT IGNORE INTO blockchain_data (txid, num_inputs, num_outputs, fee, mempool_entry_time, block_height) VALUES (%s, %s, %s, %s, %s, %s)"
                self.cursor.executemany(sql, chunk)
                self.batch_blockchain_data = self.batch_blockchain_data[chunk_size:]
                self.connection.commit()

            while self.batch_tx_inputs:
                chunk = self.batch_tx_inputs[:chunk_size]
                sql = "INSERT IGNORE INTO tx_inputs (txid, input_order, address, value) VALUES (%s, %s, %s, %s)"
                self.cursor.executemany(sql, chunk)
                self.batch_tx_inputs = self.batch_tx_inputs[chunk_size:]
                self.connection.commit()

            while self.batch_tx_outputs:
                chunk = self.batch_tx_outputs[:chunk_size]
                sql = "INSERT IGNORE INTO tx_outputs (txid, output_order, address, value) VALUES (%s, %s, %s, %s)"
                self.cursor.executemany(sql, chunk)
                self.batch_tx_outputs = self.batch_tx_outputs[chunk_size:]
                self.connection.commit()
        return self.a