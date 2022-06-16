import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope='session')
def test_arr():
    data = np.arange(3)
    assert isinstance(data, np.ndarray)
    return data


@pytest.fixture(scope='session')
def test_df():
    data = pd.DataFrame({
        'integer': [1, 2, 3],
        'square': [1, 4, 9],
        'letter': ['a', 'b', 'c'],
    })
    assert isinstance(data, pd.DataFrame)
    return data


@pytest.fixture(scope='session')
def test_dict():
    data = {
        'a': 1,
        'b': 2,
        'c': 3,
    }
    assert isinstance(data, dict)
    return data


@pytest.fixture(scope='session')
def test_list():
    data = [1, 2, 3]
    assert isinstance(data, list)
    return data


@pytest.fixture(scope='session')
def test_obj():
    class Foo:
        bar = 'bar'
    return Foo()


@pytest.fixture(scope='session')
def test_str():
    data = 'lorem ipsum dolor'
    assert isinstance(data, str)
    return data
