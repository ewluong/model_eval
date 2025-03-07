import numpy as np

def compare_models(global_metrics: dict, predictions: dict, data, target_column: str, task: str) -> dict:
    """
    Compare models based on computed metrics and prediction differences.
    
    Parameters:
        global_metrics (dict): Metrics dictionary for each model.
        predictions (dict): Predictions dictionary for each model.
        data (DataFrame): Test dataset.
        target_column (str): Target column name.
        task (str): Task type: 'classification' or 'regression'.
    
    Returns:
        dict: Summary of model comparisons including metric differences and prediction differences.
    """
    comparison_summary = {}
    comparison_summary['metrics'] = global_metrics
    
    differences = []
    model_names = list(predictions.keys())
    
    for idx in range(len(data)):
        row_diff = {}
        true_value = data[target_column].iloc[idx]
        row_diff['index'] = idx
        row_diff['true'] = true_value
        row_diff['predictions'] = {model: predictions[model][idx] for model in model_names}
        
        if task == 'classification':
            correctness = {model: (pred == true_value) for model, pred in row_diff['predictions'].items()}
            if len(set(correctness.values())) > 1:
                differences.append(row_diff)
        else:
            preds = np.array(list(row_diff['predictions'].values()))
            if np.max(preds) - np.min(preds) > 0.1 * np.mean(np.abs(preds)):
                differences.append(row_diff)
    
    comparison_summary['prediction_differences'] = differences
    
    return comparison_summary
