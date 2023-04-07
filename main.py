#Assignment: https://dlo.mijnhva.nl/d2l/lms/dropbox/user/folder_submit_files.d2l?db=207186&grpid=0&isprv=0&bp=0&ou=471098
#Assignment Guide: file:///C:/Users/Christan/Desktop/Big-Data-Engineer/Big%20Data%20Scientist%20and%20Engineer%20Individual%20Assignment%201-old%20program.pdf
import pandas as pd

# Load the CSV file into a pandas dataframe
df = pd.read_csv("Hotel_Reviews.csv")

# View the first few rows of the dataframe
print(df.head())