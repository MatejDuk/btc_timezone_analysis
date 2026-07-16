from get_address_info import GetAddressInfo
import os
import pymysql
import pandas as pd
import streamlit as st
import concurrent.futures
import time

class HeuristicClustering:
    def __init__(self, start_address, connection, cursor, session, table_placeholder):
        self.start_address = start_address
        self.connection = connection
        self.cursor = cursor
        self.session = session
        self.api_key = os.getenv("API_KEY")
        self.table_placeholder = table_placeholder

        self.new_iteration = []

        self.new_addresses = []
        self.old_addresses = []

        self.addresses = []
        self.inputs = []
        self.outputs = []
        self.blockchain = []

        self.write = pd.DataFrame(columns = ["Address", "Number of outgoing txs", "Number of incoming txs", "Address source", "Iteration"])

    def decim(self, x):
        try:
            s = str(x)
            return len(s.split(".")[1])
        except Exception:
            s = format(x, 'f').rstrip('0').rstrip('.')  
            if '.' in s:
                return len(s.split('.')[1])
            return 0
        

    def html_request(self, address, offset=0, in_out = "incoming"):
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
        start_time = time.time()
        while test:
            j += 1
            try:
                r = self.session.get(url, timeout=(10, 20), params=params, headers = headers)
                r = r.json()
                
            except Exception as e:
                print(f"Attempt {j} error on {self.address}: {e}")
                time.sleep(2) 
            else:
                break
        end_time = time.time()
        #print(end_time-start_time)
        print(address)
        return r
    
    
    def inputs_analysis(self, inputs, new_iteration):
        inputs = pd.DataFrame(inputs, columns = ["txid", "input_order", "address", "value"])
        input_txids = inputs[inputs["address"].isin(new_iteration)]["txid"].unique()

        input_addresses = inputs[inputs["txid"].isin(input_txids)]["address"].unique().tolist()
        input_addresses = list(set(input_addresses) - set(new_iteration))
        return input_addresses
                
    def change_analysis(self, inputs, outputs, blockchain_data, new_iteration):
        blockchain_data = pd.DataFrame(blockchain_data, 
                               columns = ["txid", "num_inputs", "num_outputs", "fee", "mempool_entry_time", "block_height"])
        input_data = pd.DataFrame(inputs,
                                    columns = ["txid", "input_order", "address", "value"])
        output_data = pd.DataFrame(outputs,
                                columns = ["txid", "output_order", "address", "value"])

        outgoing_txids = input_data[input_data["address"].isin(new_iteration)]["txid"].unique().tolist()
        change_addresses = []

        for txid in outgoing_txids:
            input_addresses = input_data[input_data["txid"] == txid]["address"].unique().tolist()
            output_addresses = output_data[output_data["txid"] == txid]["address"].unique().tolist()

            if set(output_addresses) & set(input_addresses):
                        continue
            
            if len(output_addresses) == 2:
                address1, address2 = output_addresses[0], output_addresses[1]

                r1 = self.html_request(address1, offset=0, in_out="incoming")
                r2 = self.html_request(address2, offset=0, in_out="incoming")
                
                val1 = output_data[(output_data["txid"] == txid) & (output_data["output_order"] == 0)]["value"]
                val2 = output_data[(output_data["txid"] == txid) & (output_data["output_order"] == 1)]["value"]

                if len(r1) == 1 and self.decim(val1)-self.decim(val2) >= 3:
                    change_addresses.append(address1)
                if len(r2) == 1 and self.decim(val2) - self.decim(val1) >= 3:
                    change_addresses.append(address2)
            elif len(output_addresses) > 2:
                count = 0
                for address in output_addresses:
                    r = self.html_request(address, offset=0,in_out="incoming")
                    if len(r) == 1:
                        cha = address
                        count +1
                if count == 1:
                    change_addresses.append(cha)
        return change_addresses

    def execute_batch_insert(self, sql, data, batch_size=500):
        """Batches data to avoid MySQL packet size limits and connection drops."""
        for i in range(0, len(data), batch_size):
            batch = data[i : i + batch_size]
            try:
                self.cursor.executemany(sql, batch)
                self.connection.commit()
            except Exception as e:
                print(f"Error at batch {i}: {e}")
                self.connection.rollback()
                raise
    
    
    def heuristic_clus(self):
        self.new_iteration = [self.start_address]
        iteration = 0 
        input_addresses = []
        change_addresses = []
        self.old_addresses = [self.start_address]

        while True:
            for address in self.new_iteration:
                new_addresses_info = GetAddressInfo(address, self.session)
                new_addresses_info.fetch_and_extract()
                self.inputs += new_addresses_info.batch_tx_inputs
                self.outputs += new_addresses_info.batch_tx_outputs
                self.blockchain += new_addresses_info.batch_blockchain_data

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
                        "Address source": ["Heuristic  clustering"],
                        "Iteration": [iteration]
                    })
                    
                st.session_state.write =pd.concat([st.session_state.write, new_row], ignore_index=True)
                self.table_placeholder.dataframe(st.session_state.write)
            iteration += 1
            


            self.new_addresses = input_addresses + change_addresses
            
            diff_addr = list(set(self.new_addresses) - set(self.old_addresses))
            if len(diff_addr) == 0:
                break
            self.new_iteration = diff_addr
            
            # 1. Blockchain Data
            sql_b = "INSERT IGNORE INTO blockchain_data (txid, num_inputs, num_outputs, fee, mempool_entry_time, block_height) VALUES (%s, %s, %s, %s, %s, %s)"
            self.execute_batch_insert(sql_b, self.blockchain)

            # 2. Inputs Data
            sql_i = "INSERT IGNORE INTO tx_inputs (txid, input_order, address, value) VALUES (%s, %s, %s, %s)"
            self.execute_batch_insert(sql_i, self.inputs)

            # 3. Outputs Data
            sql_o = "INSERT IGNORE INTO tx_outputs (txid, output_order, address, value) VALUES (%s, %s, %s, %s)"
            self.execute_batch_insert(sql_o, self.outputs)

            self.inputs = []
            self.outputs = []
            self.blockchain = []
                

            self.new_addresses = diff_addr
            self.old_addresses += diff_addr
            input_addresses = []
            change_addresses = []



        
        

        

            



    