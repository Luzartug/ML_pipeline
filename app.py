import streamlit as st
import os
import json
import requests
import sklearn
import pandas as pd
from dotenv import load_dotenv
load_dotenv()

from src.pipeline.inference_pipeline import InferencePipeline
from src.pipeline.train_pipeline import TrainingPipeline
from src.pipeline.data_pipeline import DataPipeline

openweathermap_key = os.getenv('OPEN_WEATHER_MAP_KEY')

def extract_current(latitude, longitude):
    getUrl = 'https://api.openweathermap.org/data/2.5/air_pollution'
    response = requests.get(getUrl, params={"lat": latitude, "lon": longitude, "appid": openweathermap_key})
    response=json.loads(response.content.decode('utf-8'))
    df = pd.DataFrame(response['list'])
    return pd.json_normalize(df['components'])
    

def main():
    st.title("Latitude and Longitude Selection")

    # Latitude and Longitude selection
    latitude = st.slider("Select Latitude", -90.0, 90.0, 0.0)
    longitude = st.slider("Select Longitude", -180.0, 180.0, 0.0)

    # Display the selected coordinates
    st.write(f"Selected Latitude: {latitude}")
    st.write(f"Selected Longitude: {longitude}")

    # Save button
    if st.button('Save'):
        df_features=extract_current(latitude, longitude)
        st.dataframe(df_features)
        inference_pipeline= InferencePipeline()
        st.write(inference_pipeline.inference(df_features))

    if st.button('Train Pipeline'):
        data_pipeline=DataPipeline()
        train_path, test_path = data_pipeline.data()
        
        training_pipeline=TrainingPipeline()
        training_pipeline.training(train_path, test_path)
        st.write("Pipeline trained successfully")
        
if __name__ == "__main__":
    main()