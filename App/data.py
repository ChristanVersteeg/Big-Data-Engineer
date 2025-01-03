from pymongo import MongoClient 
import pandas as pd
import col

client = MongoClient("localhost", 27017)
db = client['Big']
collection = db['Data']

# 700000
df = pd.DataFrame(list(collection.find().limit(10000)))