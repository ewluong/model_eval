import os
import pickle
import tempfile
import joblib
import pytest
from model_eval_suite import model_loader

class DummyModel:
    def predict(self, X):
        return [0] * len(X)

def test_load_model_pickle():
    model = DummyModel()
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as tmp:
        pickle.dump(model, tmp)
        tmp_path = tmp.name
    loaded_model = model_loader.load_model(tmp_path)
    assert hasattr(loaded_model, 'predict')
    os.remove(tmp_path)

def test_load_model_joblib():
    model = DummyModel()
    with tempfile.NamedTemporaryFile(delete=False, suffix='.joblib') as tmp:
        joblib.dump(model, tmp.name)
        tmp_path = tmp.name
    loaded_model = model_loader.load_model(tmp_path)
    assert hasattr(loaded_model, 'predict')
    os.remove(tmp_path)

def test_load_model_invalid_path():
    with pytest.raises(FileNotFoundError):
        model_loader.load_model("non_existent_file.pkl")

def test_load_model_no_predict(tmp_path):
    dummy_obj = object()
    file_path = tmp_path / "dummy.pkl"
    with open(file_path, "wb") as f:
        pickle.dump(dummy_obj, f)
    with pytest.raises(AttributeError):
        model_loader.load_model(str(file_path))
