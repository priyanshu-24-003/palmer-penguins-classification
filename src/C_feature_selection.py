from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import pandas as pd
import os
import logging
import yaml



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


def check_column_scores(cols):
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
    
    return cross_val_score(LR, X_train[cols], y_train, cv = 5).mean()

def FeatureSelection()


def main():
    try:

        params = load_params(params_path='params.yaml')
        max_features = params['feature_engineering']['max_features']

        train_data = load_data('./data/interim/train_processed.csv')
        test_data = load_data('./data/interim/test_processed.csv')

        # train_df, test_df = apply_tfidf(train_data, test_data, max_features)

        save_data(train_df, os.path.join("./data", "processed", "train_tfidf.csv"))
        save_data(test_df, os.path.join("./data", "processed", "test_tfidf.csv"))
    except Exception as e:
        logger.error('Failed to complete the feature engineering process: %s', e)
        print(f"Error: {e}")


if __name__ == '__main__':
    main()