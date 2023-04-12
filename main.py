#Assignment: https://dlo.mijnhva.nl/d2l/lms/dropbox/user/folder_submit_files.d2l?db=207186&grpid=0&isprv=0&bp=0&ou=471098
#Assignment Guide: file:///C:/Users/Christan/Desktop/Big-Data-Engineer/Big%20Data%20Scientist%20and%20Engineer%20Individual%20Assignment%201-old%20program.pdf
import requests
from bs4 import BeautifulSoup
from textblob import TextBlob
import time
import pandas as pd
import csv
import random

# Load the CSV file into a pandas dataframe
df = pd.read_csv("Hotel_Reviews.csv")

# View the first few rows of the dataframe
print(df.head())

# Replace this URL with the one you want to scrape reviews from
url = 'https://www.tripadvisor.com/Hotels-g187147-Paris_Ile_de_France-Hotels.html'

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.82 Safari/537.36'
}

def get_review_data(url):
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"An error occurred while fetching the URL: {e}")
        return []

    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Replace the CSS selector with the appropriate one for the website you're scraping
    reviews = soup.select('.ghulh')
    
    return [review.get_text() for review in reviews]

def classify_review_sentiment(review_text):
    analysis = TextBlob(review_text)
    if analysis.sentiment.polarity > 0:
        return 'positive'
    elif analysis.sentiment.polarity < 0:
        return 'negative'
    else:
        return 'neutral'

def main():
    review_texts = get_review_data(url)
    
    if not review_texts:
        print("No reviews found.")
        return

    classified_reviews = [(review, classify_review_sentiment(review)) for review in review_texts]

    for review, sentiment in classified_reviews:
        print(f"Review: {review.strip()}\nSentiment: {sentiment}\n")

    # Introduce a delay between requests to avoid overwhelming the server or getting blocked
    time.sleep(random.uniform(1, 3))

if __name__ == '__main__':
    main()