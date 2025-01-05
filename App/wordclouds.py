import streamlit as st
from wordcloud import WordCloud
import data
import col

def generate_and_save(text, file_name):
    wordcloud = WordCloud(
        width=800,
        height=400,
        mode="RGBA",
        background_color=None,
        colormap="viridis",
        max_words=200
    ).generate(text)

    wordcloud.to_file(file_name)

def draw():
    positive_text = " ".join(data.df[col.POSITIVE_REVIEW].dropna())
    negative_text = " ".join(data.df[col.NEGATIVE_REVIEW].dropna())

    positive_file = "positive_wordcloud.png"
    negative_file = "negative_wordcloud.png"

    generate_and_save(positive_text, file_name=positive_file)
    generate_and_save(negative_text, file_name=negative_file)

    st.subheader("Positive Reviews Word Cloud")
    st.image(positive_file)

    st.subheader("Negative Reviews Word Cloud")
    st.image(negative_file)