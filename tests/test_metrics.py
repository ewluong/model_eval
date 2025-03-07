from model_eval_suite import metrics

def test_compute_classification_metrics():
    y_true = [0, 1, 0, 1]
    y_pred = [0, 1, 1, 1]
    result = metrics.compute_classification_metrics(y_true, y_pred)
    assert 'accuracy' in result
    assert 'precision' in result
    assert 'recall' in result
    assert 'f1' in result
    assert 'confusion_matrix' in result

def test_compute_regression_metrics():
    y_true = [1.0, 2.0, 3.0]
    y_pred = [1.1, 1.9, 3.2]
    result = metrics.compute_regression_metrics(y_true, y_pred)
    assert 'mse' in result
    assert 'mae' in result
    assert 'r2' in result
