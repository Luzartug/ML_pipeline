import os
import sys
from src.exception import CustomException
from src.logger import logging

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from dataclasses import dataclass

@dataclass
class DataTransformationConfig:
    train_data_path: str=os.path.join('data', 'curated', 'train.txt')
    test_data_path: str=os.path.join('data', 'curated', 'test.txt')
    
class DataTransformation:
    def __init__(self):
        self.transformation_config=DataTransformationConfig()
        
    def transform_data(self, raw_path):
        '''
        This function is responsible for data transformation
        '''
        try:
            raw_data=pd.read_csv(raw_path)
            logging.info('Read raw data')
            
            # Keep only 'Date' and 'Close' columns
            df = raw_data[['Date', 'Close']]
            
            # Date -> to_datetime
            df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')
            
            # Filter and then delete 'Date'
            df = df[(df['Date'] > '2021-01-01') & (df['Date'] < '2023-11-28')]
            del df['Date']
            
            # Scale 'Close' column
            scaler = MinMaxScaler(feature_range=(0, 1))
            df = scaler.fit_transform(np.array(df['Close']).reshape(-1, 1))
            
            logging.info('Trasnformation completed')

            # Split data into training and test sets
            training_size = int(len(df) * 0.60)
            train_data = df[:training_size, :]
            test_data = df[training_size:, :]
            
            os.makedirs(os.path.dirname(self.transformation_config.train_data_path), exist_ok=True)
            
            # Save transformed data
            np.savetxt(self.transformation_config.train_data_path, train_data, delimiter=",")
            np.savetxt(self.transformation_config.test_data_path, test_data, delimiter=",")
            
            logging.info('Train/test split initiated')
            
            return(
                self.transformation_config.train_data_path,
                self.transformation_config.test_data_path
            )
            
        except Exception as e:
            raise CustomException(e,sys)
        
     