from get_address_info import GetAddressInfo
import os
import pymysql
import pandas as pd
import streamlit as st

class HeuristicClustering:
    def __init__(self, start_address, a, connection, cursor, session, table_placeholder):
        self.start_address = start_address
        self.a = a
        self.connection = connection
        self.cursor = cursor
        self.session = session
        self.table_placeholder = table_placeholder

        raw_ips = os.getenv("IP_ADDRESSES")
        self.IP_ADDR = [ip.strip() for ip in raw_ips.split(",")] if raw_ips else []
        self.proxies_dict_list = []

        for line in self.IP_ADDR:
            try:
                ip, port, user, pwd = line.split(":")
                self.proxies_dict_list.append({
                    "http": f"http://{user}:{pwd}@{ip}:{port}",
                    "https": f"http://{user}:{pwd}@{ip}:{port}"
                })
            except ValueError:
                pass 
            
        self.length_of_ip_a = len(self.proxies_dict_list)
        self.written_addresses = [start_address] 
        self.addresses = []

    def decim(self, x):
        try:
            s = str(x)
            return len(s.split(".")[1])
        except Exception:
            s = format(x, 'f').rstrip('0').rstrip('.')  
            if '.' in s:
                return len(s.split('.')[1])
            return 0
    
    def inputs_analysis(self, out_txids):
        if not out_txids:
            return []
            
        placeholders = ','.join(['%s'] * len(out_txids))
        query = f"SELECT DISTINCT address FROM tx_inputs WHERE txid IN ({placeholders})"
        self.cursor.execute(query, out_txids)
        
        return [x[0] for x in self.cursor.fetchall()]
        
    def change_analysis(self, out_txids):
        if not out_txids:
            return []
            
        placeholders = ','.join(['%s'] * len(out_txids))
        
        self.cursor.execute(f"SELECT DISTINCT txid, address FROM tx_inputs WHERE txid IN ({placeholders});", out_txids)
        inputs_total = pd.DataFrame(self.cursor.fetchall(), columns=["txid", "address"])

        self.cursor.execute(f"SELECT txid, output_order, address, value FROM tx_outputs WHERE txid IN ({placeholders});", out_txids)
        out_info = pd.DataFrame(self.cursor.fetchall(), columns=["txid", "output_order", "address", "value"])
        out_unique = out_info.drop_duplicates(subset=["txid", "address"])

        change_addresses = []
        
        for txid in out_txids:
            input_addresses = inputs_total[inputs_total["txid"] == txid]["address"].tolist()
            output_addresses = out_info[out_info["txid"] == txid]["address"].tolist()
            
            if set(output_addresses) & set(input_addresses):
                continue
                
            if len(output_addresses) == 2:
                address1, address2 = output_addresses[0], output_addresses[1]
                len_a1 = len(out_unique[out_unique["address"] == address1])
                len_a2 = len(out_unique[out_unique["address"] == address2])  
                
                val1 = out_info[(out_info["txid"] == txid) & (out_info["address"] == address1)]["value"].iloc[0]
                val2 = out_info[(out_info["txid"] == txid) & (out_info["address"] == address2)]["value"].iloc[0]
                
                if len_a1 == 1 and (self.decim(val1) - self.decim(val2)) >= 3:
                    change_addresses.append(address1)
                elif len_a2 == 1 and (self.decim(val2) - self.decim(val1)) >= 3:
                    change_addresses.append(address2)
                    
            elif len(output_addresses) > 2:
                output_info = out_info[out_info["txid"] == txid].copy()
                output_info["decim"] = output_info["value"].apply(self.decim)
                max_dec = output_info["decim"].max()
                
                address_candidates = output_info[output_info["decim"] == max_dec]["address"]
                
                if len_address_candidates := len(address_candidates) == 1:
                    address = address_candidates.iloc[0]
                    less_than_max = output_info[output_info["decim"] < max_dec]["decim"]
                    second_max = less_than_max.max() if not less_than_max.empty else 0
                    
                    if len(out_unique[out_unique["address"] == address]) == 1 and (max_dec - second_max) >= 3:
                        change_addresses.append(address) 
                        
        return change_addresses
    
    def heuristic_clus(self):
        self.addresses = [self.start_address]
        new_addresses = [self.start_address]
        old_txid = []
        
        iteration = 1
        
        while True:
            # 1. Gather all txids associated with the new layer of addresses in one round-trip
            placeholders = ','.join(['%s'] * len(new_addresses))
            query = f"SELECT DISTINCT txid FROM tx_inputs WHERE address IN ({placeholders})"
            self.cursor.execute(query, new_addresses)
            
            fetched_txids = [x[0] for x in self.cursor.fetchall()]
            current_txids = list(set(fetched_txids) - set(old_txid))
            
            if not current_txids:
                break
                
            input_addresses = self.inputs_analysis(current_txids)
            change_addresses = self.change_analysis(current_txids)
            
            combined_discovered = list(set(input_addresses + change_addresses))
            old_txid.extend(current_txids)
            
            new_addresses_to_scrape = []
            for addr in combined_discovered:
                if addr not in self.addresses:
                    new_addresses_to_scrape.append(addr)
                    self.addresses.append(addr)

            if not new_addresses_to_scrape:
                break

            # Use an in-memory list to accumulate data points for this iteration's batch
            batch_rows = []
            
            for addr in new_addresses_to_scrape:
                scraper = GetAddressInfo(addr, self.a, self.connection, self.cursor, self.session)
                scraper.address_write()
                self.a = scraper.a
                
                # Append raw dicts instead of converting to DataFrame immediately
                batch_rows.append({
                    "Address": addr,
                    "Number of outgoing txs": scraper.outgoing_count,
                    "Number of incoming txs": scraper.incoming_count,
                    "Address source": "Heuristic Clustering",
                    "Iteration": iteration
                })
                self.written_addresses.append(addr)

            # 2. Batch-update the UI once per iteration layer instead of per address
            if batch_rows:
                new_df = pd.DataFrame(batch_rows)
                st.session_state.first_address = pd.concat(
                    [st.session_state.first_address, new_df], 
                    ignore_index=True
                )
                self.table_placeholder.dataframe(st.session_state.first_address)

            new_addresses = new_addresses_to_scrape
            iteration += 1
            
        return self.addresses



    