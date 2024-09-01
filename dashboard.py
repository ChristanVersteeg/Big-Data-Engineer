from pymongo import MongoClient 
import pandas as pd

client = MongoClient("localhost", 27017)
db = client['Big']
collection = db['Data']

df = pd.DataFrame(list(collection.find().limit(10000)))

print(df)