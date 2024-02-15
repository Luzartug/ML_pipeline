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
            # transformation of the data
            
            return(
                train_data,
                test_data
            )
            
        except Exception as e:
            raise CustomException(e,sys)
        
     