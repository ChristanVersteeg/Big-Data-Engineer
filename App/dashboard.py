import streamlit as st

import col
import data
import totals
import piechart


st.title("Hotel Reviews Data Dashboard")
st.write("Big Data Engineer Dashboard.")

st.sidebar.header("Select columns to display")
selected_columns = st.sidebar.multiselect("", data.df.columns.tolist(), default=col.included_columns)
st.dataframe(data.df[selected_columns])

totals.draw()
piechart.draw()