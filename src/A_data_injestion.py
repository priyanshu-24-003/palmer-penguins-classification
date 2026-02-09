import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import logging
import os
from sklearn.model_selection import train_test_split


# creating a log file for later debbuging.

log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger('data_ingestion')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path = os.path.join(log_dir, 'data_ingestion.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


#data loader
def load_data(data_url: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    try:
        df = pd.read_csv(data_url)
        logger.debug('Data loaded from %s', data_url)
        return df
    
    except Exception as e:
        logger.error('Unexpected error occurred while loading the data: %s', e)
        raise


def train_test_spliter(df, test_size, random_state):
    try:
        XY, xy = train_test_split(df, test_size=test_size, random_state=random_state)
        logger.debug('split the data into training and testing')
        return (XY, xy)
    except:
        logger.error('error in train_test_spliter function')
        
def basic_processing(df):
    #Look at just the Species, Island, Culmen Length, Culmen Depth, Flipper Length, Body Mass, and Sex Columns
    df = df.drop(["studyName", "Sample Number", "Region", "Stage", "Individual ID", "Clutch Completion",
                "Date Egg", "Delta 15 N (o/oo)", "Delta 13 C (o/oo)", "Comments"], axis = 1)

    #Shorten the species names to Adelie, Gentoo, and Chinstrap
    df["Species"] = df["Species"].str.split().str.get(0)

    #Drop the one row where Sex was recorded as "."
    df = df[df["Sex"] != "."]

    #Drop rows that include NaN 
    df = df.dropna(subset = ["Sex"])

    return df

def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, data_path: str) -> None:
    """Save the train and test datasets."""
    try:
        raw_data_path = data_path
        os.makedirs(raw_data_path, exist_ok=True)
        train_data.to_csv(os.path.join(raw_data_path, "train.csv"), index=False)
        test_data.to_csv(os.path.join(raw_data_path, "test.csv"), index=False)
        logger.debug('Train and test data saved to %s', raw_data_path)
    except Exception as e:
        logger.error('Unexpected error occurred while saving the data: %s', e)
        raise

def main():    
    df = load_data('data/src/palmer_penguins.csv')
    df = basic_processing(df)

    XY, xy = train_test_spliter(df, 0.2, 42)

    save_data(XY, xy, './data/raw')


if __name__ == '__main__':
    main()
