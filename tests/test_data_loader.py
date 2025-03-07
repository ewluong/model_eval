import pandas as pd
import pytest
from model_eval_suite import data_loader

def test_load_data_success(tmp_path):
    df = pd.DataFrame({
        'feature': [1, 2, 3],
        'target': [0, 1, 0]
    })
    file_path = tmp_path / "data.csv"
    df.to_csv(file_path, index=False)
    
    loaded_df = data_loader.load_data(str(file_path), 'target')
    pd.testing.assert_frame_equal(loaded_df, df)

def test_load_data_missing_target(tmp_path):
    df = pd.DataFrame({
        'feature': [1, 2, 3],
        'label': [0, 1, 0]
    })
    file_path = tmp_path / "data.csv"
    df.to_csv(file_path, index=False)
    
    with pytest.raises(ValueError):
        data_loader.load_data(str(file_path), 'target')

def test_load_data_missing_file():
    with pytest.raises(Exception):
        data_loader.load_data("non_existent_file.csv", "target")
