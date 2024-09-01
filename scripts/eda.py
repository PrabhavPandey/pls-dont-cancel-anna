import pandas as pd
import numpy as np

def clean_numeric(x):
  if pd.api.types.is_string_dtype(x):
    return pd.to_numeric(x.str.replace(',', '').str.replace('₹', ''), errors='coerce')
  elif pd.api.types.is_numeric_dtype(x):
    return x
  else:
    return pd.to_numeric(x, errors='coerce')

# Load datasets
all_time_data = pd.read_csv('data/all_time_data.csv')
trip_trends = pd.read_csv('data/trip_trends.csv')
registration_trends = pd.read_csv('data/registration_trends.csv')

# Data cleaning and preprocessing for all_time_data
numeric_columns = ['Searches', 'Searches which got estimate', 'Searches for Quotes', 'Searches which got Quotes', 
                   'Bookings', 'Completed Trips', 'Cancelled Bookings', "Drivers' Earnings", 
                   'Average Distance per Trip (km)', 'Average Fare per Trip', 'Distance Travelled (km)']

for col in numeric_columns:
  all_time_data[col] = clean_numeric(all_time_data[col])

percentage_columns = ['Search-to-estimate Rate', 'Rider Fare Acceptance Rate', 'Driver Quote Acceptance Rate', 
                       'Quote-to-booking Rate', 'Driver Cancellation Rate',  # Updated
                       'User Cancellation Rate', 'Conversion Rate']

for col in percentage_columns:
  if pd.api.types.is_string_dtype(all_time_data[col]):
    all_time_data[col] = all_time_data[col].str.rstrip('%').astype('float') / 100.0
  elif pd.api.types.is_numeric_dtype(all_time_data[col]):
    all_time_data[col] = all_time_data[col] / 100.0

# Clean trip_trends and registration_trends
for df in [trip_trends, registration_trends]:
  df['Time'] = pd.to_datetime(df['Time'])
  df['Day'] = df['Time'].dt.day_name()
  numeric_cols = df.columns.drop(['Time', 'Day'])
  for col in numeric_cols:
    df[col] = clean_numeric(df[col])

# Save preprocessed data
all_time_data.to_csv('data/preprocessed_all_time_data.csv', index=False)
trip_trends.to_csv('data/preprocessed_trip_trends.csv', index=False)
registration_trends.to_csv('data/preprocessed_registration_trends.csv', index=False)

print("Data preprocessing completed successfully.")