import os
from jinja2 import Template
from types import SimpleNamespace

def generate_report(global_metrics: dict, subgroup_results: dict, comparison_summary: dict, charts: dict, output_dir: str) -> str:
    """
    Generate a single consolidated HTML report that embeds all results and interactive visualizations.
    
    Parameters:
        global_metrics (dict): Overall metrics for each model.
        subgroup_results (dict): Subgroup analysis results.
        comparison_summary (dict): Comparison summary including prediction differences.
        charts (dict): HTML snippets for generated charts.
        output_dir (str): Directory to save the report.
    
    Returns:
        str: Path to the generated HTML report.
    """
    comparison_obj = SimpleNamespace(**comparison_summary)
    
    html_template_str = """
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Model Evaluation & Comparison Report</title>
  <style>
    body { background-color: #111; color: #EEE; font-family: Arial, sans-serif; margin: 20px; }
    h1, h2, h3, h4 { color: #FFD700; }
    .section { margin-bottom: 40px; }
    .chart { margin-bottom: 40px; }
    .chart-title { margin-bottom: 10px; font-size: 1.2em; font-weight: bold; }
    pre { background: #222; padding: 10px; }
  </style>
  <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
</head>
<body>
  <h1>Model Evaluation & Comparison Report</h1>
  
  <div class="section">
    <h2>Executive Summary</h2>
    <p>This report summarizes the performance of the evaluated models.</p>
  </div>
  
  <div class="section">
    <h2>Global Metrics</h2>
    {% for model, metrics in global_metrics.items() %}
      <h3>{{ model }}</h3>
      <ul>
        {% for metric, value in metrics.items() %}
          <li><strong>{{ metric }}:</strong> {{ value }}</li>
        {% endfor %}
      </ul>
    {% endfor %}
  </div>
  
  <div class="section">
    <h2>Subgroup Analysis</h2>
    {% if subgroup_results %}
      {% for feature, groups in subgroup_results.items() %}
        <h3>Analysis by {{ feature }}</h3>
        {% for subgroup, metrics in groups.items() %}
          <p><strong>{{ feature }} = {{ subgroup }}:</strong></p>
          <ul>
            {% for model, model_metrics in metrics.items() %}
              <li><strong>{{ model }}:</strong>
                <ul>
                  {% for metric, value in model_metrics.items() %}
                    <li>{{ metric }}: {{ value }}</li>
                  {% endfor %}
                </ul>
              </li>
            {% endfor %}
          </ul>
        {% endfor %}
      {% endfor %}
    {% else %}
      <p>No subgroup analysis was performed.</p>
    {% endif %}
  </div>
  
  <div class="section">
    <h2>Model Comparison Summary</h2>
    <h3>Metrics Comparison</h3>
    {% for model, metrics in comparison_summary.metrics.items() %}
      <h4>{{ model }}</h4>
      <ul>
        {% for metric, value in metrics.items() %}
          <li><strong>{{ metric }}:</strong> {{ value }}</li>
        {% endfor %}
      </ul>
    {% endfor %}
    <h3>Prediction Differences</h3>
    {% if comparison_summary.prediction_differences %}
      <ul>
        {% for diff in comparison_summary.prediction_differences %}
          <li><strong>Row {{ diff.index }}:</strong> True value = {{ diff.true }}, Predictions = {{ diff.predictions }}</li>
        {% endfor %}
      </ul>
    {% else %}
      <p>No significant prediction differences detected.</p>
    {% endif %}
  </div>
  
  <div class="section">
    <h2>Visualizations</h2>
    {% for name, chart_html in charts.items() %}
      <div class="chart">
        <div class="chart-title">{{ name }}</div>
        <div>{{ chart_html | safe }}</div>
      </div>
    {% endfor %}
  </div>
  
</body>
</html>
    """
    template = Template(html_template_str)
    report_content = template.render(
        global_metrics=global_metrics,
        subgroup_results=subgroup_results,
        comparison_summary=comparison_obj,
        charts=charts
    )
    
    report_path = os.path.join(output_dir, 'model_evaluation_report.html')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
    return report_path
