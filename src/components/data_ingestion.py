import os
import sys
from src.exception import CustomException
from src.logger import logging

import pandas as pd
import numpy as np
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

@dataclass
class DataIngestionConfig:
    raw_data_path: str=os.path.join('data','raw', 'raw.csv')
    
class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()
        
    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion")
        try:
            # Read the dataset - use API
            df = pd.read_csv('data/data.csv', index_col=0)
            logging.info('Read csv data')
            
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)
            
            df.to_csv(self.ingestion_config.raw_data_path,index=False, header=True)
            
            logging.info('Ingestion completed')
            
            return(
                self.ingestion_config.raw_data_path
            )
            
        except Exception as e:
            raise CustomException(e, sys)