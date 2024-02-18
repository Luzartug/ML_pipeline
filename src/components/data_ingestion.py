import os
import sys
from src.exception import CustomException
from src.logger import logging

import pandas as pd
import numpy as np
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation

@dataclass
class DataIngestionConfig:
    raw_data_path: str=os.path.join('data','raw', 'raw.csv')
    
class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()
        
    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion")
        try:
            # Read the dataset
            df = pd.read_csv('data/AmesHousing.csv')            
            logging.info('Read csv data')
            
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)
            
            df.to_csv(self.ingestion_config.raw_data_path,index=False, header=True)
            
            logging.info('Ingestion completed')
            
            return(
                self.ingestion_config.raw_data_path
            )
            
        except Exception as e:
            raise CustomException(e, sys)
        
if __name__=="__main__":
    obj=DataIngestion()
    raw_data = obj.initiate_data_ingestion()
    data_transformation=DataTransformation()
    train_data, test_data = data_transformation.transform_data(raw_data)
    