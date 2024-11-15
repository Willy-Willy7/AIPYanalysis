import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(file_path):
    
    """
    Loads data from a CSV file.

    Parameters
    ----------
    file_path : str
        The path to the CSV file to be loaded.

    Returns
    -------
    pandas.DataFrame
        The data loaded from the CSV file.
    """
    data = pd.read_csv(file_path)
    return data

def preprocess_data(data):
    # Basic text cleaning and preprocessing
    
    data['review'] = data['review'].str.lower().str.replace('[^\w\s]', '')
    return data
    """
    Splits the given data into training and test sets.

    Parameters
    ----------
    data : pandas.DataFrame
        The data to split.

    Returns
    -------
    X_train, X_test, y_train, y_test : tuple
        A tuple containing the training and test data, split into features (X) and labels (y).
    """
def split_data(data):
    
    """
    Splits the given data into training and test sets.

    Parameters
    ----------
    data : pandas.DataFrame
        The data to split.

    Returns
    -------
    X_train, X_test, y_train, y_test : tuple
        A tuple containing the training and test data, split into features (X) and labels (y).
    """
    X = data['review']
    y = data['sentiment']
    return train_test_split(X, y, test_size=0.2, random_state=42)

