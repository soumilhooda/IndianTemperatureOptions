import pandas as pd
import os
import matplotlib.pyplot as plt

# Define a function to load and process city data
def load_and_process_city_data(file_path):
    city_df = pd.read_csv(file_path)

    # Make sure the 'Date' column is loaded as dates
    city_df['Date'] = pd.to_datetime(city_df['Date'])

    # Set the 'Date' column as the index
    city_df.set_index('Date', inplace=True)

    # Extract city name, month, and calculate Tmean
    city_name = os.path.splitext(os.path.basename(file_path))[0].split('-')[0]
    city_df['City'] = city_name
    city_df['Month'] = city_df.index.month  # Access month directly from index
    city_df['Temp Mean'] = (city_df['Temp Min'] + city_df['Temp Max']) / 2
    city_df['Temp Mean'].fillna(method='ffill', inplace=True)  
    city_df['Temp Mean'].fillna(method='bfill', inplace=True) 

    return city_df
