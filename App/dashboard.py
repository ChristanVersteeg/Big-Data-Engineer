import streamlit as st

import col
import data
import totals
import piechart
import wordclouds

st.title("Hotel Reviews Data Dashboard")
st.write("Big Data Engineer Dashboard.")

if (not data.keras and not data.torch):
    st.sidebar.header("Select columns to display")
    selected_columns = st.sidebar.multiselect("", data.df.columns.tolist(), default=col.included_columns)
    st.dataframe(data.df[selected_columns])

totals.draw()
piechart.draw_positive_negative()
piechart.draw_score_distribution()
wordclouds.draw()