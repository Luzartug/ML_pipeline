import os
import sys
from src.exception import CustomException
from src.logger import logging

import pandas as pd
import numpy as np
from dataclasses import dataclass
import mlflow
import json
import requests
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()

from src.components.data_quality import DataQuality
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

@dataclass
class DataIngestionConfig:
    raw_data_path: str=os.path.join('data','raw', 'raw.csv')
    openweathermap_key:str = os.getenv('OPEN_WEATHER_MAP_KEY')
    
class DataIngestion:
    def __init__(self):
        self.ingestion_config= DataIngestionConfig()
        mlflow.set_experiment("Data_Ingestion")
    
    def fetch_air_pollution_data(self):
        api_url = "https://api.openweathermap.org/data/2.5/air_pollution/history"
        latitude = 40.73
        longitude = -73.93
        
        start_dt = datetime(2021, 1, 1)
        end_dt = datetime(2022, 12, 31, 23, 59, 59)
        start = int(start_dt.timestamp())
        end = int(end_dt.timestamp())
        
        params = {
            'lat': latitude,
            'lon': longitude,
            'start': start,
            'end': end,
            'appid': self.ingestion_config.openweathermap_key
        }

        response = requests.get(api_url, params=params)
        if response.status_code == 200:
            logging.info("Data fetched successfully from API.")
            return json.loads(response.content.decode('utf-8'))
        else:
            logging.error(f"Failed to fetch data from API. Status code: {response.status_code}")
    
    def initiate_data_ingestion(self):
        '''
        This function is responsible for data ingestion
        '''
        
        with mlflow.start_run(run_name="Data_Ingestion_Run"):
            try:    
                # Read the dataset - use API
                data_json = self.fetch_air_pollution_data()
                logging.info("Entered the data ingestion")
                
                if data_json:
                    df = pd.json_normalize(data_json['list'])
                    new_headers = ["dt", "aqi", "co", "no", "no2", "o3", "so2", "pm2_5", "pm10", "nh3"]
                    df.columns=new_headers
                    
                    os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)
                    
                    df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
                    mlflow.log_artifact(self.ingestion_config.raw_data_path, "raw_data")
                    
                    logging.info('Data ingestion completed and saved to CSV')
                    
                    return(
                        self.ingestion_config.raw_data_path
                    )
            except Exception as e:
                raise CustomException(e, sys)
              
if __name__ == "__main__":
    data_ingestion = DataIngestion()
    raw_data = data_ingestion.initiate_data_ingestion()
    
    # data_quality_instance = DataQuality(raw_data)
    # processed_data_path = data_quality_instance.initiate_data_quality_checks()
    # print(processed_data_path)
    
    # data_trans = DataTransformation()
    # train_set, test_set = data_trans.transform_data(processed_data_path)
    # print(train_set, '\n',test_set)
    
    # model=ModelTrainer()
    # score_f1, best_model_name=model.initiate_model_trainer(train_set, test_set)
    # print(score_f1, best_model_name)