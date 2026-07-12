from get_address_info import GetAddressInfo
from heuristic_clustering import HeuristicClustering
import pymysql
import os
import requests_cache
import datetime

session = requests_cache.CachedSession('api_cache', expire_after=datetime.timedelta(days=30))

connection = pymysql.connect(
            host="gateway01.eu-central-1.prod.aws.tidbcloud.com",
            port=4000,
            user=os.getenv("USER_DATABASE"),
            password=os.getenv("PASSWORD_DATABASE"),
            database="test", 
            ssl_verify_cert=True,
            ssl_verify_identity=True
        )
cursor = connection.cursor()

start_address = "1JGzp1NqjPvyNfottDGgKFtxgjfGdyTmnU"
new_address = GetAddressInfo(start_address,0,connection, cursor, session)
new_address.address_write()
a = new_address.a

clust_scrapper = HeuristicClustering(start_address, a, connection, cursor, session)
clust_scrapper.heuristic_clus()