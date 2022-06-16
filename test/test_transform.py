from feature_transform import transform
from sklearn.base import TransformerMixin
import numpy as np
import pytest


@pytest.mark.parametrize('spec', [
    {
        'dataset': {
            'transform': {'module': 'sklearn', 'n_jobs': 1}
        },
        'transform': {
            'x': {
                'sepal length (cm)': {'StandardScaler': None},
                'sepal width (cm)': {'StandardScaler': None},
                'petal length (cm)': {'StandardScaler': None},
                'petal width (cm)': {'StandardScaler': None},
            },
            'y': {
                'target': {'OneHotEncoder': None}
            }
        }
    }
])
@pytest.mark.parametrize('stage', ['fit', 'test', 'validate'])
def test_fit_transform(spec, stage, test_data_df):
    xs, ys = transform.fit_transform(spec, stage, test_data_df)
    assert isinstance(xs, np.ndarray)
    assert isinstance(ys, np.ndarray)
    assert len(xs) == len(ys)
    artifacts = transform.get_artifacts(spec)
    for mode, col_transfmr in artifacts['mode2col_transfmr'].items():
        assert isinstance(col_transfmr, TransformerMixin)
    for mode, transformed_names in artifacts['mode2transformed_names'].items():
        assert isinstance(transformed_names, list)
        for name in transformed_names:
            assert isinstance(name, str)


@pytest.mark.parametrize('spec', [
    {
        'dataset': {
            'transform': {'module': 'sklearn', 'n_jobs': None}
        },
        'transform': {
            'x': {
                # chained-transformer in a pipeline, and custom preprocessor
                'sepal length (cm)': {'Log1pScaler': None, 'StandardScaler': None},
                'sepal width (cm)': {'Clipper': {'a_min': 0, 'a_max': 10}, 'StandardScaler': None},
                'petal length (cm)': {'StandardScaler': None},
                'petal width (cm)': {'StandardScaler': None}
            },
            'y': {
                'target': {
                    # specify kwargs to transformer
                    'OneHotEncoder': {
                        'sparse': False, 'handle_unknown': 'ignore'}
                }
            }
        }
    },
    {
        'dataset': {
            'transform': {'module': 'sklearn', 'n_jobs': None}
        },
        'transform': {
            'foo': {  # different modes
                'sepal length (cm)': {'Log1pScaler': None, 'StandardScaler': None},
                'sepal width (cm)': {'Clipper': {'a_min': 0, 'a_max': 10}, 'StandardScaler': None},
                'petal length (cm)': {'StandardScaler': None},
                'petal width (cm)': {'StandardScaler': None}
            },
            'bar': {
                'target': {
                    # specify kwargs to transformer
                    'OneHotEncoder': {
                        'sparse': False, 'handle_unknown': 'ignore'}
                }
            },
            'baz': {
                'target': {
                    'OrdinalEncoder': None
                }
            }
        }
    }
])
@pytest.mark.parametrize('stage', ['fit'])
def test_transform_spec(spec, stage, test_data_df):
    data_list = transform.fit_transform(spec, stage, test_data_df)
    for data in data_list:
        assert isinstance(data, np.ndarray)
    artifacts = transform.get_artifacts(spec)
    for mode, col_transfmr in artifacts['mode2col_transfmr'].items():
        assert isinstance(col_transfmr, TransformerMixin)
    for mode, transformed_names in artifacts['mode2transformed_names'].items():
        assert isinstance(transformed_names, list)
        for name in transformed_names:
            assert isinstance(name, str)
