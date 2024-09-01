import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Load preprocessed data
all_time_data = pd.read_csv('data/preprocessed_all_time_data.csv')
trip_trends = pd.read_csv('data/preprocessed_trip_trends.csv')

# Merge datasets
ward_day_data = trip_trends.groupby('Day')[['Searches', 'Completed Trips']].mean().reset_index()
merged_data = all_time_data.merge(ward_day_data, how='cross')

# Prepare features and target variables
X = merged_data[['Ward', 'Day', 'Average Distance per Trip (km)']]
y_cancellation = merged_data['Driver Cancellation Rate']  # Updated
y_fare = merged_data['Average Fare per Trip']

# Encode categorical variables
le_ward = LabelEncoder()
le_day = LabelEncoder()
X.loc[:, 'Ward'] = le_ward.fit_transform(X['Ward'])
X.loc[:, 'Day'] = le_day.fit_transform(X['Day'])

# Split data
X_train, X_test, y_cancel_train, y_cancel_test, y_fare_train, y_fare_test = train_test_split(
    X, y_cancellation, y_fare, test_size=0.2, random_state=42)

# Train Cancellation Prediction Model (regression)
rf_cancellation = RandomForestRegressor(n_estimators=100, random_state=42)
rf_cancellation.fit(X_train, y_cancel_train)

# Evaluate Cancellation Prediction Model
cancel_pred = rf_cancellation.predict(X_test)
cancel_mse = mean_squared_error(y_cancel_test, cancel_pred)
cancel_r2 = r2_score(y_cancel_test, cancel_pred)
print(f"Cancellation Prediction MSE: {cancel_mse:.2f}")
print(f"Cancellation Prediction R2 Score: {cancel_r2:.2f}")

# Train Fare Estimation Model (regression)
rf_fare = RandomForestRegressor(n_estimators=100, random_state=42)
rf_fare.fit(X_train, y_fare_train)

# Evaluate Fare Estimation Model
fare_pred = rf_fare.predict(X_test)
fare_mse = mean_squared_error(y_fare_test, fare_pred)
fare_r2 = r2_score(y_fare_test, fare_pred)
print(f"Fare Estimation MSE: {fare_mse:.2f}")
print(f"Fare Estimation R2 Score: {fare_r2:.2f}")

# Save models and encoders
joblib.dump(rf_cancellation, 'models/cancellation_model.joblib')
joblib.dump(rf_fare, 'models/fare_model.joblib')
joblib.dump(le_ward, 'models/ward_encoder.joblib')
joblib.dump(le_day, 'models/day_encoder.joblib')