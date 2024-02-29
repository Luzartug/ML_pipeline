import os
import sys
sys.path.append('/Users/lucaspascal/Work/S9/ML_in_prod/ML_NBA_pipeline')
import pytest
from unittest.mock import patch
from src.components.data_ingestion import DataIngestion

# Test for successful API call
@patch('src.components.data_ingestion.requests.get')
def test_fetch_air_pollution_data_success(mock_get):
    mock_response = mock_get.return_value
    mock_response.status_code = 200
    mock_response.json.return_value = {
        'list': [{'dt': 1609459200, 'components': {'co': 201.94}}]  # Simplified example data
    }
    data_ingestion = DataIngestion()
    result = data_ingestion.fetch_air_pollution_data()
    assert 'list' in result

# Test for failed API call
@patch('src.components.data_ingestion.requests.get')
def test_fetch_air_pollution_data_failure(mock_get):
    mock_response = mock_get.return_value
    mock_response.status_code = 404
    data_ingestion = DataIngestion()
    with pytest.raises(CustomException):
        data_ingestion.fetch_air_pollution_data()

# Test for data ingestion process
@patch('src.components.data_ingestion.DataIngestion.fetch_air_pollution_data')
def test_initiate_data_ingestion(mock_fetch):
    mock_fetch.return_value = {
        'list': [{'dt': 1609459200, 'components': {'co': 201.94}}]  # Simplified example data
    }
    data_ingestion = DataIngestion()
    raw_data_path = data_ingestion.initiate_data_ingestion()
    # Check if the file is created, and the content is correct
    # You can use os.path to check for file existence and pandas to read the CSV and validate content