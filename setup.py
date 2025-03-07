from setuptools import setup, find_packages

setup(
    name='model_eval_suite',
    version='0.1.0',
    description='A tool to evaluate and compare machine learning models, now with a Streamlit UI.',
    author='Your Name',
    author_email='your.email@example.com',
    packages=find_packages(),
    install_requires=[
        'pandas',
        'numpy',
        'scikit-learn',
        'matplotlib',
        'Jinja2',
        'joblib',
        'plotly',
        'streamlit'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
    ],
)
