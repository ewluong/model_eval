import os
import sys
import tempfile
import subprocess
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import joblib

def create_dummy_classification_models(temp_dir):
    iris = load_iris()
    X, y = iris.data, iris.target
    model1 = LogisticRegression(max_iter=200)
    model1.fit(X, y)
    model1_path = os.path.join(temp_dir, 'model1.pkl')
    joblib.dump(model1, model1_path)
    
    model2 = DecisionTreeClassifier()
    model2.fit(X, y)
    model2_path = os.path.join(temp_dir, 'model2.pkl')
    joblib.dump(model2, model2_path)
    
    return model1_path, model2_path

def create_dummy_dataset(temp_dir):
    from sklearn.datasets import load_iris
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['species'] = iris.target
    data_path = os.path.join(temp_dir, 'iris_test.csv')
    df.to_csv(data_path, index=False)
    return data_path

def test_cli_multiple_models():
    with tempfile.TemporaryDirectory() as temp_dir:
        model1_path, model2_path = create_dummy_classification_models(temp_dir)
        data_path = create_dummy_dataset(temp_dir)
        output_dir = os.path.join(temp_dir, 'results')
        os.makedirs(output_dir, exist_ok=True)
        
        cmd = [
            sys.executable, "-m", "streamlit", "run", "streamlit_app.py",
            "--", "--models", model1_path, model2_path,
            "--data", data_path,
            "--task", "classification",
            "--target", "species",
            "--output", output_dir
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
        
        # In a Streamlit app, integration testing is more complex. This test simply verifies the process exits successfully.
