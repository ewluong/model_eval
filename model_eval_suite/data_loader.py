import pandas as pd

def load_data(data_path: str, target_column: str) -> pd.DataFrame:
    """
    Load a CSV dataset and validate that the target column exists.
    
    Parameters:
        data_path (str): Path to CSV file.
        target_column (str): Name of target column.
    
    Returns:
        DataFrame: Loaded data with no missing values in target column.
    
    Raises:
        Exception: If file is not found, or target column is missing.
    """
    try:
        data = pd.read_csv(data_path)
    except Exception as e:
        raise Exception(f"Error reading CSV file: {e}")
    
    if target_column not in data.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataset.")
    
    data = data.dropna(subset=[target_column])
    
    return data
