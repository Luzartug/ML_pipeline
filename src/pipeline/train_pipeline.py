import os
import sys
import pandas as pd
import numpy as np
from src.exception import CustomException

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

class TrainingPipeline:
    def __init__(self):
        pass
    
    def training(self):
        try:
            obj=DataIngestion()
            raw_data = obj.initiate_data_ingestion()
            
            data_transformation=DataTransformation()
            train_path, test_path = data_transformation.transform_data(raw_data)
            
            model_trainer=ModelTrainer()
            f1_score=model_trainer.initiate_model_trainer(train_path=train_path, test_path=test_path)

        except Exception as e:
            raise CustomException(e, sys)
        
if __name__=="__main__":
    training_pipeline=TrainingPipeline()
    training_pipeline.training()