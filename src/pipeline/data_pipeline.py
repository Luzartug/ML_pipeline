import os
import sys
import pandas as pd
import numpy as np
from src.exception import CustomException

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation

class DataPipeline:
    def __init__(self):
        pass
    
    def data(self):
        try:
            obj=DataIngestion()
            raw_data=obj.initiate_data_ingestion()
            
            # Add Data quality step
            
            data_transformation=DataTransformation()
            train_path, test_path=data_transformation.transform_data(raw_data)

            return train_path, test_path
        except Exception as e:
            raise CustomException(e, sys)
        
if __name__=="__main__":
    data_pipeline=DataPipeline()
    train_path, test_path = data_pipeline.data()
    