from feature_transform import util
from pathlib import Path
import numpy as np
import os
import pandas as pd
import pydash as ps
import pytest


def test_json_encoder(test_arr, test_df, test_dict, test_list, test_obj):
    encoder = util.JsonEncoder()
    assert encoder.encode(test_arr) == '[0, 1, 2]'
    assert encoder.encode(test_df['integer']) == '[1, 2, 3]'
    assert encoder.encode(test_dict) == '{"a": 1, "b": 2, "c": 3}'
    assert encoder.encode(test_list) == '[1, 2, 3]'
    assert encoder.encode(np.array([1], dtype=np.int64)[0]) == '1'
    assert encoder.encode(np.array([1.1], dtype=np.float64)[0]) == '1.1'
    with pytest.raises(TypeError):
        encoder.encode(test_obj)


def test_abspath():
    rel_path = Path('test/test_util.py')
    abs_path = Path(__file__)
    assert util.abspath(rel_path) == abs_path
    assert util.abspath(abs_path) == abs_path
    assert util.abspath(abs_path, as_dir=True) == abs_path.parent


def test_get_file_ext():
    assert util.get_file_ext('foo.csv') == '.csv'


def test_get_spec_sha(test_dict):
    spec_sha = util.get_spec_sha(test_dict)
    assert ps.is_string(spec_sha)
    assert spec_sha == '03a7362'


def test_read_file_not_found():
    fake_rel_path = 'test/lib/test_util.py_fake'
    with pytest.raises(FileNotFoundError):
        util.read(fake_rel_path)


@pytest.mark.parametrize('filename,dtype', [
    ('test_df.csv', pd.DataFrame),
])
def test_write_read_as_df(tmpdir, test_df, filename, dtype):
    data_path = tmpdir / f'test/fixture/lib/util/{filename}'
    util.write(test_df, util.abspath(data_path))
    assert os.path.exists(data_path)
    data_df = util.read(util.abspath(data_path))
    assert isinstance(data_df, dtype)


@pytest.mark.parametrize('filename,dtype', [
    ('test_df.pkl', pd.DataFrame),
])
def test_write_read_as_pickle(tmpdir, test_df, filename, dtype):
    data_path = tmpdir / f'test/fixture/lib/util/{filename}'
    util.write(test_df, util.abspath(data_path))
    assert os.path.exists(data_path)
    data_df = util.read(util.abspath(data_path))
    assert isinstance(data_df, dtype)


@pytest.mark.parametrize('filename,dtype', [
    ('test_dict.json', dict),
    ('test_dict.yml', dict),
])
def test_write_read_as_plain_dict(tmpdir, test_dict, filename, dtype):
    data_path = tmpdir / f'test/fixture/lib/util/{filename}'
    util.write(test_dict, util.abspath(data_path))
    assert os.path.exists(data_path)
    data_dict = util.read(util.abspath(data_path))
    assert isinstance(data_dict, dtype)


@pytest.mark.parametrize('filename,dtype', [
    ('test_list.json', list),
])
def test_write_read_as_plain_list(tmpdir, test_list, filename, dtype):
    data_path = tmpdir / f'test/fixture/lib/util/{filename}'
    util.write(test_list, util.abspath(data_path))
    assert os.path.exists(data_path)
    data_dict = util.read(util.abspath(data_path))
    assert isinstance(data_dict, dtype)


@pytest.mark.parametrize('filename,dtype', [
    ('test_str.txt', str),
])
def test_write_read_as_plain_str(tmpdir, test_str, filename, dtype):
    data_path = tmpdir / f'test/fixture/lib/util/{filename}'
    util.write(test_str, util.abspath(data_path))
    assert os.path.exists(data_path)
    data_dict = util.read(util.abspath(data_path))
    assert isinstance(data_dict, dtype)
