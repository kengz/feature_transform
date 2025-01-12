import pandas as pd
import polars as pl
import pytest
import sklearn
from sklearn import datasets

import feature_transform as ft
from feature_transform.helper import suggest_sr


def test_suggest():
    x_df, y_sr = datasets.load_wine(return_X_y=True, as_frame=True)

    # suggest spec_dict - use directly or save to yaml for further editing
    spec_dict = ft.suggest(x_df)
    assert isinstance(spec_dict, dict)
    assert "preprocessing.RobustScaler" in str(spec_dict)
    assert "preprocessing.PowerTransformer" in str(spec_dict)

    col_tfm = ft.build(spec_dict)
    assert isinstance(col_tfm, sklearn.compose.ColumnTransformer)


def test_suggest_type_error():
    # Test that a TypeError is raised when input is not a pandas or polars DataFrame/Series
    with pytest.raises(TypeError):
        suggest_sr("invalid_input")

    with pytest.raises(TypeError):
        suggest_sr(42)


def test_suggest_missing_values():
    # Test that SimpleImputer is suggested for columns with missing values (using pandas)
    data = {"col1": [1, 2, None, 4], "col2": [1, None, 3, 4]}
    df = pd.DataFrame(data)
    spec_dict = ft.suggest(df)

    # Check if SimpleImputer is suggested for columns with missing values
    assert "impute.SimpleImputer" in str(spec_dict)


def test_suggest_onehotencoder():
    # Test that OneHotEncoder is suggested for categorical columns (using pandas)
    data = {"col1": ["cat", "dog", "cat", "dog"], "col2": [1, 2, 3, 4]}
    df = pd.DataFrame(data)
    spec_dict = ft.suggest(df)

    # Check if OneHotEncoder is suggested for 'col1'
    assert "preprocessing.OneHotEncoder" in str(spec_dict)


def test_suggest_standard_scaler():
    # Test that StandardScaler is suggested for normal numeric columns without outliers (using pandas)
    data = {"col1": [1, 2, 3, 4], "col2": [10, 20, 30, 40]}
    df = pd.DataFrame(data)
    spec_dict = ft.suggest(df)

    # Check if StandardScaler is suggested for columns without skew or outliers
    assert "preprocessing.StandardScaler" in str(spec_dict)


def test_suggest_with_polars():
    # Test that the suggest function works with polars DataFrame
    data = {"col1": ["cat", "dog", "cat", "dog"], "col2": [1, 2, 3, 4]}
    df = pl.DataFrame(data)
    spec_dict = ft.suggest(df)

    # Check if OneHotEncoder is suggested for 'col1'
    assert "preprocessing.OneHotEncoder" in str(spec_dict)


def test_suggest_with_missing_values_polars():
    # Test that SimpleImputer is suggested for missing values in polars DataFrame
    data = {"col1": [1, 2, None, 4], "col2": [1, None, 3, 4]}
    df = pl.DataFrame(data)
    spec_dict = ft.suggest(df)

    # Check if SimpleImputer is suggested for columns with missing values
    assert "impute.SimpleImputer" in str(spec_dict)


def test_suggest_for_skewed_data():
    # Test that PowerTransformer is suggested for skewed data (using pandas)
    data = {"col1": [1, 2, 3, 1000], "col2": [1, 2, 3, 4]}
    df = pd.DataFrame(data)
    spec_dict = ft.suggest(df, skew_threshold=1.0)

    # Check if PowerTransformer is suggested for 'col1' which is skewed
    assert "preprocessing.PowerTransformer" in str(spec_dict)


def test_suggest_for_normal_data():
    # Test that StandardScaler is suggested for normal data (using pandas)
    data = {"col1": [1, 2, 3, 4], "col2": [10, 20, 30, 40]}
    df = pd.DataFrame(data)
    spec_dict = ft.suggest(df, skew_threshold=1.0)

    # Check if StandardScaler is suggested for normal data
    assert "preprocessing.StandardScaler" in str(spec_dict)
