from pathlib import Path
from typing import Any
import dill
import hashlib
import json
import numpy as np
import os
import pandas as pd
import pydash as ps
import yaml
try:
    import hydra
except ImportError:
    pass


class JsonEncoder(json.JSONEncoder):
    '''Add numpy and pandas types to json serialization format'''

    def default(self, obj: Any) -> str:
        if isinstance(obj, (np.ndarray, pd.Series)):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        else:
            raise TypeError('Unrecognized new type for JSON serialization')


def abspath(data_path: str, as_dir: bool = False) -> Path:
    '''
    Resolve any relative data_path into abspath relative to DIR

    @example
    ```
    util.abspath('slm_lab/lib')
    # => '/Users/ANON/Documents/slm_lab/slm_lab/lib'

    util.abspath('/tmp')
    # => '/tmp'
    ```
    '''
    data_path = Path(data_path)  # guard
    if not data_path.is_absolute():
        dir = Path(os.getenv('DIR', Path.cwd()))
        data_path = dir / data_path
    if as_dir and data_path.is_file():
        data_path = data_path.parent
    return data_path.resolve()


def cfg_to_dict(cfg) -> dict:
    '''Convert hydra config to dict to allow dict operations'''
    return hydra.utils.instantiate(cfg, _convert_='all')  # convert to dict


def get_cfg(config_dir: str, config_name: str = 'config.yaml', overrides: list = []):
    '''
    Convenience method to get hydra config outside of @hydra
    @example
    from pathlib import Path
    from feature_transformer import util

    DIR = Path(__file__).parent
    cfg = util.get_cfg(DIR / 'config')
    '''
    with hydra.initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        cfg = hydra.compose(config_name, overrides=overrides)
    return cfg


def get_file_ext(data_path: str) -> str:
    '''Get the `.ext` of file.ext'''
    return Path(data_path).suffix


def get_spec_sha(spec: dict) -> str:
    '''Calculate sha by serializing spec'''
    # clone and serialize all values as str
    cast_spec = ps.clone_deep(spec)
    for k, v in cast_spec.items():
        cast_spec[k] = str(v)
    # calculate sha1, trim to first 7 chars
    spec_sha = hashlib.sha1(json.dumps(cast_spec, sort_keys=True).encode()).hexdigest()[:7]
    return spec_sha


def read(data_path: str, **kwargs) -> Any:
    '''
    Universal data reading method with smart data parsing

    - `.csv` to DataFrame
    - `.json` to dict, list
    - `.yml|.yaml` to dict
    - else to str

    @example
    ```
    data_df = util.read('test/fixture/lib/util/test_df.csv')
    # => <DataFrame>

    data_dict = util.read('test/fixture/lib/util/test_dict.json')
    data_dict = util.read('test/fixture/lib/util/test_dict.yml')
    # => <dict>

    data_list = util.read('test/fixture/lib/util/test_list.json')
    # => <list>

    data_str = util.read('test/fixture/lib/util/test_str.txt')
    # => <str>
    ```
    '''
    data_path = abspath(data_path)
    try:
        assert data_path.is_file()
    except AssertionError:
        raise FileNotFoundError(data_path)
    ext = get_file_ext(data_path)
    if ext == '.csv':
        data = read_as_df(data_path, **kwargs)
    elif ext == '.pkl':
        data = read_as_pickle(data_path, **kwargs)
    else:
        data = read_as_plain(data_path, **kwargs)
    return data


def read_as_df(data_path: str, **kwargs) -> pd.DataFrame:
    '''Submethod to read data as DataFrame'''
    data = pd.read_csv(data_path, **kwargs)
    return data


def read_as_pickle(data_path: str, **kwargs) -> Any:
    '''Submethod to read data as pickle'''
    with open(data_path, 'rb') as f:
        data = dill.load(f)
    return data


def read_as_plain(data_path: str, **kwargs) -> Any:
    '''Submethod to read data as plain type'''
    open_file = open(data_path, 'r')
    ext = get_file_ext(data_path)
    if ext == '.json':
        data = json.load(open_file, **kwargs)
    elif ext in ('.yml', '.yaml'):
        data = yaml.load(open_file, Loader=yaml.FullLoader, **kwargs)
    else:
        data = open_file.read()
    open_file.close()
    return data


def write(data: Any, data_path: str, **kwargs) -> str:
    '''
    Universal data writing method with smart data parsing

    - `.csv` from DataFrame
    - `.json` from dict, list
    - `.yml|.yaml` from dict
    - else from str(*)

    @returns The data path written to
    @example
    ```
    data_path = util.write(data_df, 'test/fixture/lib/util/test_df.csv')

    data_path = util.write(data_dict, 'test/fixture/lib/util/test_dict.json')
    data_path = util.write(data_dict, 'test/fixture/lib/util/test_dict.yml')

    data_path = util.write(data_list, 'test/fixture/lib/util/test_list.json')

    data_path = util.write(data_str, 'test/fixture/lib/util/test_str.txt')
    ```
    '''
    data_path = abspath(data_path)
    data_path.parent.mkdir(parents=True, exist_ok=True)
    ext = get_file_ext(data_path)
    if ext == '.csv':
        write_as_df(data, data_path, **kwargs)
    elif ext == '.pkl':
        write_as_pickle(data, data_path, **kwargs)
    else:
        write_as_plain(data, data_path, **kwargs)
    return data_path


def write_as_df(data: Any, data_path: str, **kwargs) -> str:
    '''Submethod to write data as DataFrame'''
    df = pd.DataFrame(data) if isinstance(data, pd.DataFrame) else data
    df.to_csv(data_path, **kwargs)
    return data_path


def write_as_pickle(data: Any, data_path: str, **kwargs) -> str:
    '''Submethod to write data as pickle'''
    with open(data_path, 'wb') as f:
        dill.dump(data, f, **kwargs)
    return data_path


def write_as_plain(data: Any, data_path: str, **kwargs) -> str:
    '''Submethod to write data as plain type'''
    open_file = open(data_path, 'w')
    ext = get_file_ext(data_path)
    if ext == '.json':
        json.dump(data, open_file, indent=2, cls=JsonEncoder, **kwargs)
    elif ext in ('.yml', '.yaml'):
        yaml.dump(data, open_file, **kwargs)
    else:
        open_file.write(str(data), **kwargs)
    open_file.close()
    return data_path
