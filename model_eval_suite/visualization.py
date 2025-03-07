import plotly.express as px
import plotly.graph_objects as go

def generate_visualizations(global_metrics: dict, task: str, data, target_column: str, predictions: dict, probabilities: dict):
    # If there are no global metrics, return an empty dict to avoid StopIteration errors.
    if not global_metrics:
        return {}
    
    charts = {}
    # Create interactive bar charts for overall metric comparison (excluding non-numeric ones)
    example_model = next(iter(global_metrics))
    metric_names = [k for k in global_metrics[example_model].keys() if k != 'confusion_matrix']
    
    for metric in metric_names:
        fig = px.bar(
            x=list(global_metrics.keys()),
            y=[global_metrics[model].get(metric) or 0 for model in global_metrics.keys()],
            labels={'x': 'Model', 'y': metric},
            title=f'Model Comparison: {metric}',
            template='plotly_dark'
        )
        charts[metric] = fig
    
    if task == 'classification':
        # Confusion matrices as interactive heatmaps
        for model_name, metrics_dict in global_metrics.items():
            cm = metrics_dict.get('confusion_matrix')
            if cm:
                fig = go.Figure(data=go.Heatmap(z=cm, colorscale='Viridis'))
                fig.update_layout(
                    title=f'Confusion Matrix: {model_name}',
                    xaxis_title='Predicted',
                    yaxis_title='True',
                    template='plotly_dark'
                )
                charts[f'confusion_matrix_{model_name}'] = fig
        # ROC curves for binary classification if probabilities are provided
        if data[target_column].nunique() == 2:
            from sklearn.metrics import roc_curve, auc
            for model_name, proba in probabilities.items():
                if proba is not None:
                    fpr, tpr, _ = roc_curve(data[target_column], proba[:, 1])
                    roc_auc = auc(fpr, tpr)
                    fig = px.line(
                        x=fpr,
                        y=tpr,
                        title=f'ROC Curve: {model_name} (AUC: {roc_auc:.2f})',
                        labels={'x': 'False Positive Rate', 'y': 'True Positive Rate'},
                        template='plotly_dark'
                    )
                    fig.add_shape(
                        type='line',
                        x0=0, y0=0, x1=1, y1=1,
                        line=dict(dash='dash', color='white')
                    )
                    charts[f'roc_curve_{model_name}'] = fig
    if task == 'regression':
        y_true = data[target_column]
        for model_name, y_pred in predictions.items():
            residuals = y_true - y_pred
            fig = px.scatter(
                x=y_pred,
                y=residuals,
                title=f'Residual Plot: {model_name}',
                labels={'x': 'Predicted Values', 'y': 'Residuals'},
                template='plotly_dark'
            )
            fig.add_hline(y=0, line_dash="dash", line_color="red")
            charts[f'residual_plot_{model_name}'] = fig
    
    return charts
