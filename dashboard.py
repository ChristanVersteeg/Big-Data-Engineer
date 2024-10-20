from pymongo import MongoClient 
import pandas as pd
import streamlit as st
import col

client = MongoClient("localhost", 27017)
db = client['Big']
collection = db['Data']

df = pd.DataFrame(list(collection.find().limit(10000)))

st.title("Hotel Reviews Data Dashboard")
st.write("Big Data Engineer Dashboard.")

st.sidebar.header("Select columns to display")
selected_columns = st.sidebar.multiselect("", df.columns.tolist(), default=col.included_columns)
st.dataframe(df[selected_columns])
