from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import pandas as pd
import os
import logging
import yaml
import numpy as np


# Ensure the "logs" directory exists
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger('C_Feature_selection')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path = os.path.join(log_dir, 'C_Feature_selection.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def load_data(file_path: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        df.fillna('', inplace=True)
        logger.debug('Data loaded and NaNs filled from %s', file_path)
        return df
    except pd.errors.ParserError as e:
        logger.error('Failed to parse the CSV file: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error occurred while loading the data: %s', e)
        raise


def check_column_scores(cols, df):
    """
    Trains and evaluates Logistic Regression via crossvalidation on the columns
    of the dataset with select indeces
    
    Parameters
    ----------
    cols : list of strings, columns on which to be trained
    
    Return
    ----------
    float : average of 5 cross validation scores
    """
    
    #Logistic Regression Model
    LR = LogisticRegression(max_iter = 5000)
    
    df2 = df.copy()
    y = df['Species']
    X = df2.drop(["Species"], axis = 1)

    return cross_val_score(LR, X[cols], y, cv = 5).mean()

def FeatureSelection(df):
    try:
        logger.debug('feature selection started ')
        quals = ["Island", "Sex"]

        quants = ["Culmen Length (mm)", "Culmen Depth (mm)",
                "Flipper Length (mm)", "Body Mass (g)"]

        combos = [[qual]+[quant1]+[quant2] for qual in quals for quant1 in quants for quant2 in quants if quant1 != quant2]

        cv_scores = []
        best_cv_score = -np.inf

        for combo in combos:
            score = check_column_scores(combo, df)
            cv_scores.append(score)
            
            if cv_scores[-1] > best_cv_score:
                best_cv_score = cv_scores[-1]
                best_combo = combo

        logger.debug("Best Feature selected which produces CV score: " + str(best_cv_score))

        return best_combo
    except:
        logger.error('error occured while selecting best featueres')
        raise
    pass



def save_data(df: pd.DataFrame, file_path: str) -> None:
    """Save the dataframe to a CSV file."""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        df.to_csv(file_path, index=False)
        logger.debug('Data saved to %s', file_path)
    except Exception as e:
        logger.error('Unexpected error occurred while saving the data: %s', e)
        raise


def main():
    try:

        # params = load_params(params_path='params.yaml')
        # max_features = params['feature_engineering']['max_features']

        train_data = load_data('./data/interim/train_processed.csv')
        test_data = load_data('./data/interim/test_processed.csv')

        BestFeatures = FeatureSelection(train_data) + ['Species']

        train_df = train_data[BestFeatures]
        test_df = test_data[BestFeatures]

        save_data(train_df, os.path.join("./data", "processed", "train_final.csv"))
        save_data(test_df, os.path.join("./data", "processed", "test_final.csv"))
    except Exception as e:
        logger.error('Failed to complete the feature engineering process: %s', e)
        print(f"Error: {e}")


if __name__ == '__main__':
    main()