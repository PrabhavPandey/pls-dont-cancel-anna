import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

# Load models and encoders
rf_cancellation = joblib.load('models/cancellation_model.joblib')
rf_fare = joblib.load('models/fare_model.joblib')
le_ward = joblib.load('models/ward_encoder.joblib')
le_day = joblib.load('models/day_encoder.joblib')
logo = Image.open('./assets/nammayatri.png')

# Load data for visualizations
all_time_data = pd.read_csv('data/preprocessed_all_time_data.csv')

tab1, tab2, tab3 = st.tabs(["⚙️Estimation Tool", "✍️About", "👋Get in Touch"])

with tab1:
    # Main App
    st.image(logo)

    # Input fields
    ward = st.selectbox('Select Origin Area (just start typing)', options=le_ward.classes_)
    day = st.selectbox('Select Day', options=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
    distance = st.number_input('Enter Ride Distance (km)', min_value=0.1, max_value=100.0, value=5.0)

    # Encode inputs
    ward_encoded = le_ward.transform([ward])[0]
    day_encoded = le_day.transform([day])[0]

    # Make predictions
    input_data = pd.DataFrame([[ward_encoded, day_encoded, distance]], columns=['Ward', 'Day', 'Average Distance per Trip (km)'])
    estimated_fare = rf_fare.predict(input_data)[0]
    cancellation_rate = rf_cancellation.predict(input_data)[0]
    # Display results
    st.header('Predictions')
    st.write(f'Estimated Fare: ₹{estimated_fare:.2f}')
    st.write(f'Estimated Driver Cancellation Rate: {cancellation_rate:.2%}')
    st.divider()

    # Line Graph for Cancellation Rates by Ward
    st.subheader('Driver Cancellation Rates by Ward')

    # Identify wards with the highest and lowest cancellation rates
    sorted_data = all_time_data[['Ward', 'Driver Cancellation Rate']]
    sort_by = 'Driver Cancellation Rate'
    sorted_data = sorted_data.sort_values(by=sort_by)
    lowest_ward = sorted_data.iloc[0:1]
    highest_ward = sorted_data.iloc[-1:]

    # Calculate the average cancellation rate
    average_cancellation_rate = sorted_data[sort_by].mean()
    average_ward = pd.DataFrame({'Ward': ['Average'], 'Driver Cancellation Rate': [average_cancellation_rate]})

    wards_to_plot = pd.concat([lowest_ward, highest_ward, average_ward, all_time_data[all_time_data['Ward'] == ward]])

    # Ensure the selected ward is in between the lowest and highest by sorting the DataFrame
    wards_to_plot = wards_to_plot.sort_values(by=sort_by).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(data=wards_to_plot, x='Ward', y=sort_by, marker='o', ax=ax)

    # Highlight the selected ward, lowest, highest, and average wards
    selected_ward_rate = all_time_data[all_time_data['Ward'] == ward]['Driver Cancellation Rate'].values[0]
    ax.plot(ward, selected_ward_rate, marker='o', markersize=10, color='red', label=f'Selected Ward: {ward}')

    lowest_ward_name = lowest_ward['Ward'].values[0]
    lowest_ward_rate = lowest_ward[sort_by].values[0]
    ax.plot(lowest_ward_name, lowest_ward_rate, marker='o', markersize=10, color='blue', label=f'Lowest: {lowest_ward_name}')

    highest_ward_name = highest_ward['Ward'].values[0]
    highest_ward_rate = highest_ward[sort_by].values[0]
    ax.plot(highest_ward_name, highest_ward_rate, marker='o', markersize=10, color='green', label=f'Highest: {highest_ward_name}')

    average_ward_rate = average_cancellation_rate
    ax.plot('Average', average_ward_rate, marker='o', markersize=10, color='orange', label='Average')

    ax.legend()
    plt.xticks(rotation=45)
    st.pyplot(fig)
    
    st.divider()

    # Historical data for selected ward
    st.subheader(f'Historical Data for {ward}')
    ward_data = all_time_data[all_time_data['Ward'] == ward].iloc[0]
    st.write(f"Average Fare: ₹{ward_data['Average Fare per Trip']:.2f}")
    st.write(f"Average Distance: {ward_data['Average Distance per Trip (km)']:.2f} km")
    st.write(f"Driver Cancellation Rate: {ward_data['Driver Cancellation Rate']:.2%}")

with tab2:
    st.title("About")
    st.write("An exploratory data analysis project on the Namma Yatri Open dataset.")
    st.image(logo, width=800)
    st.markdown("[Namma Yatri Open Dataset](https://nammayatri.in/open?cc=BLR&rides=All&tl=at)", unsafe_allow_html=True)
    st.write("Date range used for analysis: November 2022 to August 31st, 2024 (All time)")
    

with tab3:
    st.title("Get in Touch")
    st.write("Feel free to reach out to me for any queries or feedback.")
    
    # Create two columns for side-by-side buttons
    col1, col2 = st.columns(2)
    
    with col1:
        st.link_button("🧔🏻‍♂️Portfolio website", "https://prabhav.vercel.app/")
        
    with col2:
        st.link_button("🔗LinkedIn - Profile", "https://www.linkedin.com/in/prabhav-pandey/")