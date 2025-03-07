import pandas as pd
from sklearn.datasets import load_iris, load_diabetes
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier
import joblib

def create_classification_models():
    # Load Iris dataset and train two classification models
    iris = load_iris()
    X, y = iris.data, iris.target

    # Model 1: Logistic Regression
    model1 = LogisticRegression(max_iter=200)
    model1.fit(X, y)
    joblib.dump(model1, 'iris_model.pkl')
    print("Classification model saved as iris_model.pkl")

    # Model 2: Decision Tree Classifier
    model2 = DecisionTreeClassifier()
    model2.fit(X, y)
    joblib.dump(model2, 'iris_model_dt.pkl')
    print("Classification model saved as iris_model_dt.pkl")

    # Create a dummy test dataset and save it as CSV
    df = pd.DataFrame(X, columns=iris.feature_names)
    df['species'] = y  # target column
    df.to_csv('iris_test.csv', index=False)
    print("Test dataset saved as iris_test.csv")

def create_regression_model():
    # Load Diabetes dataset and train a linear regression model
    diabetes = load_diabetes()
    X, y = diabetes.data, diabetes.target
    model = LinearRegression()
    model.fit(X, y)
    joblib.dump(model, 'diabetes_model.pkl')
    print("Regression model saved as diabetes_model.pkl")
    
    # Create a dummy test dataset and save it as CSV
    df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
    df['target'] = y  # target column
    df.to_csv('diabetes_test.csv', index=False)
    print("Test dataset saved as diabetes_test.csv")

if __name__ == '__main__':
    create_classification_models()
    create_regression_model()
