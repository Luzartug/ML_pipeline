import os
import sys
import pandas as pd
import numpy as np
from src.exception import CustomException

from src.pipeline.data_pipeline import DataPipeline
from src.components.model_trainer import ModelTrainer

class TrainingPipeline:
    def __init__(self):
        pass
    
    def training(self, train_path, test_path):
        try:
            
            model_trainer=ModelTrainer()
            f1_score=model_trainer.initiate_model_trainer(train_path=train_path, test_path=test_path)

        except Exception as e:
            raise CustomException(e, sys)
        
if __name__=="__main__":
    data_pipeline=DataPipeline()
    train_path, test_path = data_pipeline.data()
    
    training_pipeline=TrainingPipeline()
    training_pipeline.training(train_path, test_path)