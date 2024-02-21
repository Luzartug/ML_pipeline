import os
import sys
from src.exception import CustomException
from src.logger import logging
from src.utils import preprocess_data

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from src.utils import save_object

from dataclasses import dataclass

@dataclass
class DataTransformationConfig:
    train_data_path: str=os.path.join('data', 'curated', 'train.csv')
    test_data_path: str=os.path.join('data', 'curated', 'test.csv')
    scaler_file_path: str=os.path.join("data", "scaler.pkl")
    
class DataTransformation:
    def __init__(self):
        self.transformation_config=DataTransformationConfig()
        
    def transform_data(self, raw_path):
        '''
        This function is responsible for data transformation
        '''
        logging.info("Entered the data transformation")
        try:
            # transformation of the data
            df = pd.read_csv('data/raw/raw.csv')            
            logging.info('Read csv data')
            
            os.makedirs(os.path.dirname(self.transformation_config.train_data_path), exist_ok=True)
        
            df, scaler = preprocess_data(df)
            
            # Save scaler
            save_object(
                file_path=self.transformation_config.scaler_file_path,
                obj=scaler
            )
            logging.info('Scaler sucessfully loaded')
            
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
            logging.info("Train/test split")
            
            train_set.to_csv(self.transformation_config.train_data_path)
            test_set.to_csv(self.transformation_config.test_data_path)
            logging.info("Train/Test csv created in ")
            
            return(
                self.transformation_config.train_data_path,
                self.transformation_config.test_data_path
            )
            
        except Exception as e:
            raise CustomException(e,sys)
        
     