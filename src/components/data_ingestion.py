import os
import sys
import requests
from datetime import datetime
from src.exception import CustomException
from src.logger import logging

import pandas as pd
from dataclasses import dataclass

from dotenv import load_dotenv
load_dotenv()

@dataclass
class DataIngestionConfig:
    raw_data_path: str = os.path.join('data', 'raw', 'raw.csv')
    lat: float  # Latitude
    lon: float  # Longitude

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.ingestion_config = config
        self.api_key = os.getenv('OPEN_WEATHER_MAP_KEY')  

    def to_unix_time(self, dt):
        """Convertit un objet datetime en timestamp UNIX."""
        return int(dt.timestamp())

    def fetch_air_pollution_data(self, start, end):
        api_url = "https://api.openweathermap.org/data/2.5/air_pollution/history"
        params = {
            'lat': self.ingestion_config.lat,
            'lon': self.ingestion_config.lon,
            'start': start,
            'end': end,
            'appid': self.api_key
        }
        response = requests.get(api_url, params=params)
        if response.status_code == 200:
            logging.info("Data fetched successfully from API.")
            return response.json()
        else:
            logging.error(f"Failed to fetch data from API. Status code: {response.status_code}")
            return None

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion process.")
        try:
            start_dt = datetime(2021, 1, 1)
            end_dt = datetime(2022, 12, 31, 23, 59, 59)
            start = self.to_unix_time(start_dt)
            end = self.to_unix_time(end_dt)

            data_json = self.fetch_air_pollution_data(start, end)
            if data_json:
                df = pd.json_normalize(data_json['list'])
                os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)
                df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
                logging.info('Data ingestion completed and saved to CSV.')
                return self.ingestion_config.raw_data_path
        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    config = DataIngestionConfig(
        lat=40.73,  
        lon=-73.93  
    )
    data_ingestion = DataIngestion(config)
    data_ingestion.initiate_data_ingestion()