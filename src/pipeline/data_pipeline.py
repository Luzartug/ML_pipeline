import os
import sys
import pandas as pd
import numpy as np
from src.exception import CustomException

from src.components.data_ingestion import DataIngestion
from src.components.data_quality import DataQuality
from src.components.data_transformation import DataTransformation

class DataPipeline:
    def __init__(self):
        pass
    
    def data(self):
        try:
            data_ingestion=DataIngestion()
            raw_data_path=data_ingestion.initiate_data_ingestion()
            
            data_quality_instance=DataQuality(raw_data_path)
            processed_data_path=data_quality_instance.initiate_data_quality_checks()
            
            data_transformation=DataTransformation()
            train_path, test_path=data_transformation.transform_data(processed_data_path)

            return train_path, test_path
        
        except Exception as e:
            raise CustomException(e, sys)
        
if __name__=="__main__":
    data_pipeline=DataPipeline()
    train_path, test_path = data_pipeline.data()
    