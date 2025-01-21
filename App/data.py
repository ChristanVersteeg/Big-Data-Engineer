from pymongo import MongoClient
import pandas as pd

db_driven = True

if db_driven:
    client = MongoClient("localhost", 27017)
    db = client['Big']
    collection = db['Data']
    
    df = pd.DataFrame(list(collection.find().limit(10000)))
    
    client.close()
else:
    csv_file = r"C:\Users\pooti\Desktop\Big-Data-Engineer\App\Hotel_Reviews.csv"
    df = pd.read_csv(csv_file)
    
def upload(data, collection_name):
    records = data.to_dict('records')
    
    for record in records:
        record.pop('_id', None)

    client = MongoClient("localhost", 27017)
    db = client['Big']
    target_coll = db[collection_name]
    
    target_coll.insert_many(records)
    
    client.close()