import pickle
import joblib
import os

def load_model(model_path: str):
    """
    Load a machine learning model from a file using pickle or joblib.
    
    Parameters:
        model_path (str): Path to the model file.
        
    Returns:
        model: Loaded model object.
    
    Raises:
        Exception: If model cannot be loaded or does not have a predict() method.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file {model_path} does not exist.")
    
    try:
        # Try loading with joblib first
        model = joblib.load(model_path)
    except Exception:
        # Fallback to pickle
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
    
    if not hasattr(model, 'predict'):
        raise AttributeError("Loaded model does not have a predict() method.")
    
    return model
