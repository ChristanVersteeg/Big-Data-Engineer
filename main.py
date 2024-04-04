#Assignment: https://dlo.mijnhva.nl/d2l/lms/dropbox/user/folder_submit_files.d2l?db=249019&grpid=0&isprv=0&bp=0&ou=540323
#Assignment Guide: file:///C:/Users/Christan/Desktop/Big-Data-Engineer/Big%20Data%20Scientist%20and%20Engineer%20Individual%20Assignment%201-old%20program.pdf
from textblob import TextBlob
import time
import pandas as pd
from enum import Enum
from sqlalchemy import create_engine

from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options

import sys #Set the encoding for standard output to UTF-8, to prevent UnicodeEncodeError(s)
if not sys.stdout.encoding or sys.stdout.encoding.lower() != 'utf-8': sys.stdout.reconfigure(encoding='utf-8')

class Type(Enum):
    CUSTOM  = 1
    KAGGLE  = 2     
    SCRAPED = 3
    
class Sentiment(Enum):
    NULL     = 1
    POSITIVE = 2     
    NEUTRAL  = 3
    NEGATIVE = 4

def setup_custom_reviews():
    custom_reviews = {
            Type: [Type.CUSTOM, Type.CUSTOM, Type.CUSTOM],
            'Review': [
                "This hotel was a damn mess. The bed sheets weren't made, the food was horrid and the staff wasn't helpful nor nice. I would not recommend.", 
                "This is your everyday average hotel, not bad, not good, reasonably priced. All in all, rather happy with how it turned out", 
                "Man this hotel was so damn amazing, and it wasn't even that expensive! It's surprising how nice the staff is, how clean the rooms are, and above all, how tasty the damn food is. I would highly recommend anyone in the proximity to go check this hotel out!"],
            'Sentiment': [Sentiment.NULL, Sentiment.NULL, Sentiment.NULL]}
    
    return custom_reviews

def setup_kaggle_reviews():
    kaggle_reviews = pd.read_csv("C:/Users/Christan/Desktop/Big-Data-Engineer/Hotel_Reviews.csv")
    kaggle_concat = pd.concat([kaggle_reviews['Positive_Review'].head(50), kaggle_reviews['Negative_Review'].head(50)], axis=0)
    
    num_reviews = len(kaggle_concat)

    kaggle_reviews = {
            Type: [Type.KAGGLE] * num_reviews,
            'Review': kaggle_concat,
            'Sentiment': [Sentiment.NULL] * num_reviews}
    
    return kaggle_reviews

def setup_scraped_reviews():
    vpn = Options()
    vpn.add_extension("C:/Users/Christan/Desktop/Big-Data-Engineer/VPN.crx")

    chrome = webdriver.Chrome(options=vpn, service=Service("C:/Users/Christan/Desktop/Big-Data-Engineer/chromedriver.exe"))
    chrome.get('https://www.tripadvisor.com/Hotels-g187147-Paris_Ile_de_France-Hotels.html')

    # 25 seconds of sleep to perform manual actions such as activating the VPN.
    time.sleep(25)

    soup = BeautifulSoup(chrome.page_source, 'html.parser')
    scraped_reviews = soup.find_all(class_='EcHSb')  # EcHSb is the recurring class name of the reviews on TripAdvisor.
    
    scraped_review_texts = [review.text for review in scraped_reviews]
    num_reviews = len(scraped_review_texts) 

    scraped_reviews = {
        Type: [Type.SCRAPED] * num_reviews,
        'Review': scraped_review_texts,
        'Sentiment': [Sentiment.NULL] * num_reviews
    }

    chrome.quit()
    
    return scraped_reviews

all_reviews = pd.concat([
    pd.DataFrame(setup_custom_reviews()),
    pd.DataFrame(setup_kaggle_reviews()),
    pd.DataFrame(setup_scraped_reviews())
], ignore_index=True)

def determine_sentiment(text):
    analysis = TextBlob(text)
    if analysis.sentiment.polarity > 0:
        return Sentiment.POSITIVE.name
    elif analysis.sentiment.polarity < 0:
        return Sentiment.NEGATIVE.name
    else:
        return Sentiment.NEUTRAL.name 
all_reviews['Sentiment'] = all_reviews['Review'].apply(determine_sentiment)
    
engine = create_engine(f'mysql+pymysql://root:BigData@127.0.0.1:3306/bigdataengineer')
all_reviews.to_sql('all_reviews', con=engine, if_exists='replace', index=False)