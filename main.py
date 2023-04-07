#Assignment: https://dlo.mijnhva.nl/d2l/lms/dropbox/user/folder_submit_files.d2l?db=207186&grpid=0&isprv=0&bp=0&ou=471098
#Assignment Guide: file:///C:/Users/Christan/Desktop/Big-Data-Engineer/Big%20Data%20Scientist%20and%20Engineer%20Individual%20Assignment%201-old%20program.pdf
import pandas as pd
import requests
from bs4 import BeautifulSoup
import csv

# Load the CSV file into a pandas dataframe
df = pd.read_csv("Hotel_Reviews.csv")

# View the first few rows of the dataframe
print(df.head())

# URL of the page you want to scrape
url = "https://www.tripadvisor.com/Hotels-g187147-Paris_Ile_de_France-Hotels.html"

# Make a GET request to the URL
response = requests.get(url)

# Parse the HTML content of the page
soup = BeautifulSoup(response.content, "html.parser")

# Find the elements you want to extract
hotels = soup.find_all("div", class_="ui_column is-8 main_col allowEllipsis")

# Create a CSV file to store the extracted data
with open("tripadvisor_hotels.csv", "w", encoding="utf-8", newline="") as csvfile:
    writer = csv.writer(csvfile)

    # Write the header row to the CSV file
    writer.writerow(["Hotel Name", "Location", "Price", "Rating"])

    # Loop through the hotels and extract the relevant information
    for hotel in hotels:
        name = hotel.find("a", class_="_15_ydu6b").get_text().strip()
        location = hotel.find("span", class_="_21qUqkJx").get_text().strip()
        price = hotel.find("div", class_="_3hDPbqJU _3hmsP_6p").get_text().strip()
        rating = hotel.find("span", class_="ui_bubble_rating").get("alt").split()[0]

        # Write the extracted information to the CSV file
        writer.writerow([name, location, price, rating])
