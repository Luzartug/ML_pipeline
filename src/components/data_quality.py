import os
import sys

from src.exception import CustomException
from src.logger import logging

import pandas as pd
import numpy as np
from scipy import stats
from dataclasses import dataclass

@dataclass
class DataQualityConfig:
    processed_data_path: str = os.path.join('data','pre-curated', 'data_quality.csv')

class DataQuality:
    def __init__(self, input_data_path):
        self.input_data_path = input_data_path
        self.quality_config = DataQualityConfig()
        self.corrections_made = False  # To track if any corrections are made

    def check_missing_values(self, df):
        missing_values = df.isnull().sum()
        total_missing = missing_values.sum()
        if total_missing > 0:
            logging.warning(f"Missing Values Found: \n{missing_values[missing_values > 0]}")
            df.fillna(method='ffill', inplace=True)  # Correction for missing values
            self.corrections_made = True
        else:
            logging.info("No missing values found in the dataset.")

    def detect_outliers(self, df):
        z_scores = np.abs(stats.zscore(df.select_dtypes(include=[np.number])))
        outliers = (z_scores > 3)
        if outliers.any().any():
            logging.warning(f"Outliers detected in {outliers.sum().sum()} rows.")
            df = df[(z_scores < 3).all(axis=1)]  # Correction for outliers
            self.corrections_made = True
        else:
            logging.info("No outliers detected in the dataset.")

    def validate_data_ranges(self, df):
        if df['aqi'].min() < 0 or df['aqi'].max() > 500:
            logging.warning("AQI values are outside the acceptable range of 0-500.")
            df['aqi'] = df['aqi'].clip(0, 500)  # Correction for AQI range
            self.corrections_made = True
        else:
            logging.info("AQI values are within the acceptable range.")

    def ensure_data_types(self, df):
        initial_count = df['dt'].isnull().sum()
        df['dt'] = pd.to_datetime(df['dt'], errors='coerce')
        after_count = df['dt'].isnull().sum()
        if after_count > initial_count:
            logging.warning("Some datetime conversions failed.")
            self.corrections_made = True
        else:
            logging.info("All datetime conversions successful.")

    def initiate_data_quality_checks(self):
        logging.info("Starting data quality checks.")
        
        try:
            df = pd.read_csv(self.input_data_path)

            self.check_missing_values(df)
            self.detect_outliers(df)
            self.validate_data_ranges(df)
            self.ensure_data_types(df)

            if self.corrections_made:
                os.makedirs(os.path.dirname(self.quality_config.processed_data_path), exist_ok=True)
                df.to_csv(self.quality_config.processed_data_path, index=False, header=True)
                logging.info("Data quality checks completed and cleaned data saved.")
                return self.quality_config.processed_data_path
            else:
                logging.info("No corrections made, original data is retained.")
                return None

        except Exception as e:
            raise CustomException(e, sys)