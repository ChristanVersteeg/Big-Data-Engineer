from pymongo import MongoClient
import pandas as pd

db_driven = False

if db_driven:
    client = MongoClient("localhost", 27017)
    db = client['Big']
    collection = db['Data']
    
    df = pd.DataFrame(list(collection.find().limit(10000)))
else:
    csv_file = r"C:\Users\pooti\Desktop\Big-Data-Engineer\App\Hotel_Reviews.csv"
    df = pd.read_csv(csv_file)