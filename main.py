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
    KAGGLE  = 2     
    SCRAPED = 3
    
class Sentiment(Enum):
    NULL     = 1
    POSITIVE = 2     
    NEUTRAL  = 3
    NEGATIVE = 4

custom_reviews = pd.DataFrame
kaggle_reviews = pd.DataFrame
scraped_reviews = pd.DataFrame

def setup_custom_reviews():
    global custom_reviews
    
    custom_reviews = {
            Type:[Type.CUSTOM, Type.CUSTOM, Type.CUSTOM],
            'Review':[
                "This hotel was a damn mess. The bed sheets weren't made, the food was horrid and the staff wasn't helpful nor nice. I would not recommend.", 
                "This is your everyday average hotel, not bad, not good, reasonably priced. All in all, rather happy with how it turned out", 
                "Man this hotel was so damn amazing, and it wasn't even that expensive! It's surprising how nice the staff is, how clean the rooms are, and above all, how tasty the damn food is. I would highly recommend anyone in the proximity to go check this hotel out!"],
            Sentiment:[Sentiment.NULL, Sentiment.NULL, Sentiment.NULL]}
setup_custom_reviews()

def setup_kaggle_reviews():
    global kaggle_reviews
    
    kaggle_reviews = pd.read_csv("C:/Users/Christan/Desktop/Big-Data-Engineer/Hotel_Reviews.csv")
    kaggle_concat = pd.concat([kaggle_reviews['Positive_Review'], kaggle_reviews['Negative_Review']], axis=0)
    
    num_reviews = len(kaggle_concat)

    kaggle_reviews = {
            Type:[Type.KAGGLE] * num_reviews,
            'Review': kaggle_concat,
            Sentiment:[Sentiment.NULL] * num_reviews}
setup_kaggle_reviews()

def setup_scraped_reviews():
    global scraped_reviews
    
    vpn = Options()
    vpn.add_extension("C:/Users/Christan/Desktop/Big-Data-Engineer/VPN.crx")

    chrome = webdriver.Chrome(options=vpn, service=Service("C:/Users/Christan/Desktop/Big-Data-Engineer/chromedriver.exe"))
    chrome.get('https://www.tripadvisor.com/Hotels-g187147-Paris_Ile_de_France-Hotels.html')

    # 25 seconds of sleep to perform manual actions such as activating the VPN. Yes this could be relatively easily automated, but do not have the time for that.
    time.sleep(25)

    soup = BeautifulSoup(chrome.page_source, 'html.parser')
    scraped_reviews = soup.find_all(class_='EcHSb') # EcHSb is the class name of the reviews.

    for review in scraped_reviews:
      print(review.text)  

    chrome.quit()
setup_scraped_reviews()

all_reviews = pd.concat(custom_reviews, kaggle_reviews, scraped_reviews)

#engine = create_engine(f'mysql+pymysql://root:BigData@127.0.0.1:3306/bigdataengineer')
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