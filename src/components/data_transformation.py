import os
import sys
from src.exception import CustomException
from src.logger import logging

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from dataclasses import dataclass


@dataclass
class DataTransformationConfig:
    train_data_path: str=os.path.join('data', 'curated', 'train.txt')
    test_data_path: str=os.path.join('data', 'curated', 'test.txt')
    
class DataTransformation:
    def __init__(self):
        self.transformation_config=DataTransformationConfig()
    
    def transform_data(self, df):
        # Drop the unnecessary columns
        columns_to_drop = ['Order', 'PID']
        X = df.drop(columns_to_drop, axis=1)
        
        # Define the numerical and categorical columns
        numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
        categorical_cols = X.select_dtypes(include=['object']).columns

        # Define the numerical and categorical transformers
        numerical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='None')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))
        ])

        # Combine transformers into a ColumnTransformer
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_cols),
                ('cat', categorical_transformer, categorical_cols)
            ])
        
        # Apply the preprocessing pipeline to the dataset
        X_processed = preprocessor.fit_transform(X)

        # The result is a numpy array, so we should convert it to a DataFrame
        # Get the list of one-hot encoded column names from the pipeline
        ohe_columns = preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(categorical_cols)

        # Combine numerical and one-hot encoded column names
        processed_columns = numerical_cols.tolist() + ohe_columns.tolist()

        # Create the processed DataFrame
        X_processed_df = pd.DataFrame(X_processed, columns=processed_columns)

        # Return the processed DataFrame
        return X_processed_df




        
     