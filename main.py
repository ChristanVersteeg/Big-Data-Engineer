#Assignment: https://dlo.mijnhva.nl/d2l/lms/dropbox/user/folder_submit_files.d2l?db=249019&grpid=0&isprv=0&bp=0&ou=540323
#Assignment Guide: file:///C:/Users/Christan/Desktop/Big-Data-Engineer/Big%20Data%20Scientist%20and%20Engineer%20Individual%20Assignment%201-old%20program.pdf
import requests
from bs4 import BeautifulSoup
from textblob import TextBlob
import time
import pandas as pd
import random
from enum import Enum
from sqlalchemy import create_engine

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager

import sys #Set the encoding for standard output to UTF-8, to prevent UnicodeEncodeError(s)
if not sys.stdout.encoding or sys.stdout.encoding.lower() != 'utf-8': sys.stdout.reconfigure(encoding='utf-8')

class Type(Enum):
    CUSTOM  = 1
    CSV     = 2     
    SCRAPED = 3
    
class Sentiment(Enum):
    NULL     = 1
    POSITIVE = 2     
    NEUTRAL  = 3
    NEGATIVE = 4
    
custom_reviews = {
        Type:[Type.CUSTOM, Type.CUSTOM, Type.CUSTOM],
        'Review':[
            "This hotel was a damn mess. The bed sheets weren't made, the food was horrid and the staff wasn't helpful nor nice. I would not recommend.", 
            "This is your everyday average hotel, not bad, not good, reasonably priced. All in all, rather happy with how it turned out", 
            "Man this hotel was so damn amazing, and it wasn't even that expensive! It's surprising how nice the staff is, how clean the rooms are, and above all, how tasty the damn food is. I would highly recommend anyone in the proximity to go check this hotel out!"],
        Sentiment:[Sentiment.NULL, Sentiment.NULL, Sentiment.NULL]}

csv = pd.read_csv("Hotel_Reviews.csv")
csv_concat = pd.concat([csv['Positive_Review'], csv['Negative_Review']], axis=0)
num_reviews = len(csv_concat)

csv_reviews = {
        Type:[Type.CSV] * num_reviews,
        'Review': csv_concat,
        Sentiment:[Sentiment.NULL] * num_reviews}

# Set up Chrome options
chrome_options = Options()
# Uncomment the next line if you want to run Chrome headless
# chrome_options.add_argument("--headless")

url = 'https://www.tripadvisor.com/Hotels-g187147-Paris_Ile_de_France-Hotels.html'
# Set up the driver
chrome = webdriver.Chrome(options=chrome_options)

# The URL you want to open

# Navigate to the page
chrome.get(url)

time.sleep(10)
# Wait for the page to load, or add explicit waits here
#chrome.implicitly_wait(10)

# Use the provided XPath to retrieve the reviews
scraped_reviews = chrome.find_elements(By.XPATH, '//*[@id="hotel-listing-2"]/div/div/div[2]/div[2]/div/div[2]/div[1]/div/div/a/span')

for review in scraped_reviews:
    print(review.text)  # Print the text of each review element found

# Close the driver
chrome.quit()


all_reviews = pd.DataFrame(custom_reviews, csv_reviews, scraped_reviews)

# Create SQL connection engine
#engine = create_engine(f'mysql+pymysql://root:BigData@127.0.0.1:3306/bigdataengineer')

# Store DataFrame in MySQL
#all_reviews.to_sql('table_name', con=engine, if_exists='replace', index=False)

#for index, row in all_reviews.iterrows():
#    review_text = row['Review']
#    analysis = TextBlob(review_text)
#    
#    if analysis.sentiment.polarity > 0:
#        sentiment = Sentiment.POSITIVE
#    elif analysis.sentiment.polarity < 0:
#        sentiment = Sentiment.NEGATIVE
#    else:
#        sentiment = Sentiment.NEUTRAL
#    
#    all_reviews.at[index, 'Sentiment'] = sentiment
