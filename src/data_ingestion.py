import pandas as pd
import os
from sklearn.model_selection import train_test_split
import logging
import yaml


# Creates a logs/ folder if it doesn’t exist, exist_ok=True means no error if folder already exists
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)


# logging configuration
# Creates a named logger (data_ingestion) Sets logging level to DEBUG (captures everything)
logger = logging.getLogger('data_ingestion')
logger.setLevel('DEBUG')

# Console logging (prints logs to terminal)
console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

# File logging (writes logs to file)
log_file_path = os.path.join(log_dir, 'data_ingestion.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

# Now logs go to: logs/data_ingestion.log

# Log format -- Example log: 2026-02-01 10:20:12 - data_ingestion - DEBUG - Data loaded from URL
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# Attach handlers to logger
# Every log message appears: In terminal and in log file
logger.addHandler(console_handler)
logger.addHandler(file_handler)

# reading configuration from YAML
def load_params(params_path: str) -> dict:
    """Load parameters from a YAML file."""
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logger.debug('Parameters retrieved from %s', params_path)
        return params
    except FileNotFoundError:
        logger.error('File not found: %s', params_path)
        raise
    except yaml.YAMLError as e:
        logger.error('YAML error: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error: %s', e)
        raise

# loading the dataset
def load_data(data_url: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    try:
        df = pd.read_csv(data_url)
        logger.debug('Data loaded successfully from %s', data_url)
        return df
    except pd.errors.ParserError as e:
        logger.error('Failed to parse the CSV file: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error occurred while loading the data: %s', e)
        raise

# data cleaning
def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the data."""
    try:
        df.drop(columns = ['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], inplace = True)
        df.rename(columns = {'v1': 'target', 'v2': 'text'}, inplace = True)
        logger.debug('Data preprocessing completed successfully')
        return df
    except KeyError as e:
        logger.error('Missing column in the dataframe: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error during preprocessing: %s', e)
        raise

# save processed data
def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, data_path: str) -> None:
    """Save the train and test datasets."""
    try:
        raw_data_path = os.path.join(data_path, 'raw')
        os.makedirs(raw_data_path, exist_ok=True)
        train_data.to_csv(os.path.join(raw_data_path, "train.csv"), index=False)
        test_data.to_csv(os.path.join(raw_data_path, "test.csv"), index=False)
        logger.debug('Train and Test data successfully saved to %s', raw_data_path)
    except Exception as e:
        logger.error('Unexpected error occurred while saving the data: %s', e)
        raise

# pipeline orchestration - This is the entry point of the pipeline.
def main():
    try:
        #params = load_params(params_path='params.yaml')
        #test_size = params['data_ingestion']['test_size']
        test_size = 0.2
        data_path = 'https://raw.githubusercontent.com/varshith-mohan/ML-Pipeline/refs/heads/main/spam.csv'
        df = load_data(data_url=data_path)
        final_df = preprocess_data(df)
        train_data, test_data = train_test_split(final_df, test_size=test_size, random_state=2)
        save_data(train_data, test_data, data_path='./data')
    except Exception as e:
        logger.error('Failed to complete the data ingestion process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()