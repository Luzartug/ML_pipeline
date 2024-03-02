import os
import sys

import pandas as pd
import pickle
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException

# Preprocess the function
def preprocess_data(df, scaler=None):
    try:
        # Normalize the pollutant columns (except 'aqi') using MinMaxScaler
        pollutants = ['co', 'no', 'no2', 'o3', 'so2', 'pm2_5', 'pm10', 'nh3']
        
        # If a scaler is not provided, initialize one
        if scaler is None:
            scaler = MinMaxScaler()
            # Fit the scaler on the data and transform the data
            df[pollutants] = scaler.fit_transform(df[pollutants])
            return df, scaler
        else:
            return scaler.transform(df)
    
    except Exception as e:
        raise CustomException(e, sys)

def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    try:
        report = {}
        
        for i in range(len(list(models))):
            model = list(models.values())[i]
            para=param[list(models.keys())[i]]
            
            # Choose the best parameter from params
            gs = GridSearchCV(model,para,cv=3)
            gs.fit(X_train,y_train)
            
            # Fit with the best model
            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)
            
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            
            train_model_score = f1_score(y_train, y_train_pred, average='weighted')
            test_model_score = f1_score(y_test, y_test_pred, average='weighted')
            
            report[list(models.keys())[i]] = test_model_score
        
        return report
    
    except Exception as e:
        raise CustomException(e, sys)

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)

def load_object(file_path):
    try:
        with open(file_path,"rb") as file_obj:
            return pickle.load(file_obj)
    
    except Exception as e:
        raise CustomException(e, sys)