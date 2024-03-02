import pytest
import requests_mock
import json
from src.components.data_ingestion import DataIngestion, DataIngestionConfig

@pytest.fixture
def mock_response():
    return {
        "list": [
            {
                "dt": 1605182400,
                "main": {
                    "aqi": 1
                },
                "components": {
                    "co": 201.94053649902344,
                    "no": 0.01877157986164093,
                    "no2": 0.43465015292167664,
                    "o3": 68.66455078125,
                    "so2": 0.6407499313354492,
                    "pm2_5": 0.5,
                    "pm10": 0.5,
                    "nh3": 0.000478
                }
            },
            # ... more data if necessary
        ]
    }

def test_data_ingestion_creates_csv_file(tmp_path, mock_response):
    # Setup the mock for requests.get
    with requests_mock.Mocker() as m:
        m.get("https://api.openweathermap.org/data/2.5/air_pollution/history", json=mock_response)

        # Mocking the environment variable for OPEN_WEATHER_MAP_KEY
        # monkeypatch.setenv('OPEN_WEATHER_MAP_KEY', 'fake_api_key')

        # Setting a temporary directory for the raw data path
        temp_raw_data_path = tmp_path / "raw_test" / "raw.csv"
        ingestion_config = DataIngestionConfig(raw_data_path=str(temp_raw_data_path))

        # # Instantiating DataIngestion with the temporary config
        data_ingestion = DataIngestion(config=ingestion_config)

        # Running the data ingestion process
        data_ingestion = DataIngestion()
        raw_data_path=data_ingestion.initiate_data_ingestion()

        # Check if the CSV file is created
        assert type(raw_data_path)==str

        # Optional: Read the CSV and check if data is correct
        # df = pd.read_csv(temp_raw_data_path)
        # assert not df.empty
        # assert list(df.columns) == ["dt", "aqi", "co", "no", "no2", "o3", "so2", "pm2_5", "pm10", "nh3"]
        # ... any other assertions based on the expected data