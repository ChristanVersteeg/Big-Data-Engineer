#Assignment: https://dlo.mijnhva.nl/d2l/lms/dropbox/user/folder_submit_files.d2l?db=207186&grpid=0&isprv=0&bp=0&ou=471098
#Assignment Guide: file:///C:/Users/Christan/Desktop/Big-Data-Engineer/Big%20Data%20Scientist%20and%20Engineer%20Individual%20Assignment%201-old%20program.pdf
import requests
from bs4 import BeautifulSoup
from textblob import TextBlob
import time
import pandas as pd
import random
from enum import Enum
import sys; sys.stdout.reconfigure(encoding='utf-8') #Set the encoding for standard output to UTF-8, to prevent UnicodeEncodeError(s)

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

pd.DataFrame(csv_reviews).head()

url = 'https://www.scrapethissite.com/pages/simple/'
response = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})

soup = BeautifulSoup(response.text, 'html.parser')
scraped_reviews = soup.find_all("div", attrs={"class":"country-info"})

reviews_text = []
for review in scraped_reviews:
    text = review.get_text(strip=True)
    print(text)
    reviews_text.append(text)

print(reviews_text)

all_reviews = pd.DataFrame(custom_reviews, csv_reviews, scraped_reviews)

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

print(all_reviews)