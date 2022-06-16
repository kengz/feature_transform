from sklearn import datasets
import pandas as pd
import pytest


@pytest.fixture(scope='session')
def test_data_df():
    # columns: sepal length (cm), sepal width (cm), petal length (cm), petal width (cm), target
    return pd.concat(datasets.load_iris(return_X_y=True, as_frame=True), axis=1)
