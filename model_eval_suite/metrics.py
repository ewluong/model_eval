from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def compute_classification_metrics(y_true, y_pred):
    """
    Compute classification metrics.
    
    Parameters:
        y_true (array-like): True labels.
        y_pred (array-like): Predicted labels.
    
    Returns:
        dict: Dictionary with accuracy, precision, recall, F1, ROC-AUC, and confusion matrix.
    """
    metrics = {}
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    metrics['recall'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    metrics['f1'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    try:
        metrics['roc_auc'] = roc_auc_score(y_true, y_pred, multi_class='ovr')
    except Exception:
        metrics['roc_auc'] = None
    
    metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred).tolist()
    
    return metrics

def compute_regression_metrics(y_true, y_pred):
    """
    Compute regression metrics.
    
    Parameters:
        y_true (array-like): True values.
        y_pred (array-like): Predicted values.
    
    Returns:
        dict: Dictionary with MSE, MAE, and R².
    """
    metrics = {}
    metrics['mse'] = mean_squared_error(y_true, y_pred)
    metrics['mae'] = mean_absolute_error(y_true, y_pred)
    metrics['r2'] = r2_score(y_true, y_pred)
    
    return metrics
