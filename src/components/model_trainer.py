import os
import sys
from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass
from src.utils import save_object, evaluate_models

import pandas as pd
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier
)
from sklearn.svm import SVC
from sklearn.metrics import f1_score

@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str=os.path.join("data", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        
    def initiate_model_trainer(self, train_path, test_path):
        try:
            train_set = pd.read_csv(train_path)
            test_set = pd.read_csv(test_path)
            logging.info("Read Data")
            
            X_train, X_test, y_train, y_test =(
                train_set[["co", "no", "no2", "o3", "so2", "pm2_5", "pm10", "nh3"]],
                test_set[["co", "no", "no2", "o3", "so2", "pm2_5", "pm10", "nh3"]],
                train_set['aqi'],
                test_set[['aqi']]
            )
            logging.info("Split training and test input data")
            
            models= {
                "Random Forest": RandomForestClassifier(),
                "Gradient Boosting": GradientBoostingClassifier(),
                "Ada Boost": AdaBoostClassifier(),
                "SVC": SVC()
            }
            params={
                "Random Forest": {
                    'n_estimators': [100]  # Number of trees in the forest
                    # 'max_features': ['auto', 'sqrt'],  # Number of features to consider at every split
                    # 'max_depth': [10, 20, 30, None],  # Maximum number of levels in tree
                    # 'min_samples_split': [2, 5, 10],  # Minimum number of samples required to split a node
                    # 'min_samples_leaf': [1, 2, 4],  # Minimum number of samples required at each leaf node
                    # 'bootstrap': [True, False]  # Method of selecting samples for training each tree
                },
                "Gradient Boosting": {
                    'n_estimators': [100]  # Number of boosting stages to be run
                    # 'learning_rate': [0.01, 0.1, 0.2],  # Learning rate shrinks the contribution of each tree
                    # 'max_depth': [3, 5, 7],  # Maximum depth of the individual regression estimators
                    # 'min_samples_split': [2, 4, 6],  # Minimum number of samples required to split a node
                    # 'min_samples_leaf': [1, 2, 4],  # Minimum number of samples required at each leaf node
                    # 'subsample': [0.8, 0.9, 1.0]  # Fraction of samples to be used for fitting the individual base learners
                },
                "Ada Boost": {
                    'n_estimators': [50]  # Maximum number of estimators at which boosting is terminated
                    # 'learning_rate': [0.01, 0.1, 1.0],  # Weight applied to each classifier at each boosting iteration
                    # 'algorithm': ['SAMME', 'SAMME.R']  # Algorithm to use
                },
                "SVC": {
                    'C': [0.1]  # Regularization parameter
                    # 'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],  # Specifies the kernel type to be used in the algorithm
                    # 'gamma': ['scale', 'auto'],  # Kernel coefficient
                    # 'degree': [2, 3, 4]  # Degree of the polynomial kernel function (‘poly’)
                }
            }
            
            model_report:dict=evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models, param=params)
            
            # Get best model f1 scores from dict
            best_model_score = max(sorted(model_report.values()))
            
            # Get best model name from dict
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]
            
            if best_model_score<0.95:
                raise CustomException("No best model found")
            logging.info("Best model found on train & test dataset")
            
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            
            predicted=best_model.predict(X_test)
            score_f1 = f1_score(y_test, predicted, average='weighted')
            return score_f1, best_model_name
        
        except Exception as e:
            raise CustomException(e, sys)