from pymongo import MongoClient
import pandas as pd

db_driven = True
keras = False
torch = False

if db_driven:
    client = MongoClient("localhost", 27017)
    db = client['Big']
    
    if(keras):
        collection = db['Keras']
    elif(torch):
        collection = db['Torch']
    else:
        collection = db['Data']
    df = pd.DataFrame(list(collection.find().limit(10000)))
    sample = pd.DataFrame(list(collection.aggregate([{"$sample": {"size": 100}}])))
    
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