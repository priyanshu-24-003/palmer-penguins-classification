from sklearn import preprocessing
import os
import logging
import pandas as pd

# creating a log file for later debbuging.
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

# Setting up logger
logger = logging.getLogger('B_preparation')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path = os.path.join(log_dir, 'B_preparation.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)




def prep_penguins_data(data, trORte):
    """
    Prepares penguins data frame for machine learning models.
    
    Parameters
    ----------
    data : pandas.DataFrame to be prepared
    
    Return
    ----------
    (X, y) : pandas.DataFrame without Species column, pandas.Series of Species column
    """
    
    #Copy data frame
    df = data.copy()
    
    try:
        #Create LabelEncoder
        le = preprocessing.LabelEncoder()
        
        #Encode Sex column in data frame
        df["Sex"] = le.fit_transform(df["Sex"])
        
        #Encode Island column in data frame
        df["Island"] = le.fit_transform(df["Island"])
        
        df["Species"] = le.fit_transform(df["Species"])

        logger.debug(f'successfully encoded the catagorical features of {trORte}')

        return df
        
    except:
        
        logger.error('error in prep_penguins_data')
        raise 
    #Create X, where X is the data frame without the Species column
    #X serves as the predictor variables
    # X = df.drop(["Species"], axis = 1)
    
    # #Create y, where y just contains the Species column
    # #y serves as the target variable
    # y = df["Species"]



def main(text_column='text', target_column='target'):
    """
    Main function to load raw data, preprocess it, and save the processed data.
    """
    try:
        # Fetch the data from data/raw
        train_data = pd.read_csv('./data/raw/train.csv')
        test_data = pd.read_csv('./data/raw/test.csv')
        logger.debug('Data loaded properly')

        # Transform the data
        train_processed_data = prep_penguins_data(train_data, 'train data')
        test_processed_data = prep_penguins_data(test_data,'test data')

        # Store the data inside data/processed
        data_path = os.path.join("./data", "interim")
        os.makedirs(data_path, exist_ok=True)
        
        train_processed_data.to_csv(os.path.join(data_path, "train_processed.csv"), index=False)
        test_processed_data.to_csv(os.path.join(data_path, "test_processed.csv"), index=False)
        
        logger.debug('Processed data saved to %s', data_path)
    except FileNotFoundError as e:
        logger.error('File not found: %s', e)
    except pd.errors.EmptyDataError as e:
        logger.error('No data: %s', e)
    except Exception as e:
        logger.error('Failed to complete the data transformation process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()
