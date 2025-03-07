import pandas as pd
from model_eval_suite import comparer

def test_compare_models_classification():
    data = pd.DataFrame({
        'target': [0, 1, 0, 1]
    })
    global_metrics = {
        'model1': {'accuracy': 0.75, 'precision': 0.8, 'recall': 0.75, 'f1': 0.77, 'roc_auc': 0.85},
        'model2': {'accuracy': 0.5, 'precision': 0.6, 'recall': 0.5, 'f1': 0.55, 'roc_auc': 0.65}
    }
    predictions = {
        'model1': [0, 1, 0, 1],
        'model2': [1, 1, 0, 0]
    }
    summary = comparer.compare_models(global_metrics, predictions, data, 'target', 'classification')
    assert 'metrics' in summary
    assert 'prediction_differences' in summary
    assert len(summary['prediction_differences']) > 0

def test_compare_models_regression():
    data = pd.DataFrame({
        'target': [1.0, 2.0, 3.0, 4.0]
    })
    global_metrics = {
        'model1': {'mse': 0.1, 'mae': 0.2, 'r2': 0.95},
        'model2': {'mse': 0.3, 'mae': 0.4, 'r2': 0.90}
    }
    predictions = {
        'model1': [1.1, 2.0, 3.0, 4.0],
        'model2': [1.5, 2.5, 3.5, 4.5]
    }
    summary = comparer.compare_models(global_metrics, predictions, data, 'target', 'regression')
    assert 'metrics' in summary
    assert 'prediction_differences' in summary
