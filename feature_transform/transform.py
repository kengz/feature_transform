from functools import partial
from loguru import logger
from feature_transform import util
from pathlib import Path
from sklearn.base import TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import FunctionTransformer
from typing import Union
import numpy as np
import pandas as pd
import pydash as ps
import sklearn.compose
import sklearn.feature_extraction
import sklearn.preprocessing


# custom data transformers


def squeeze_df(df: pd.DataFrame) -> np.array:
    return np.array(df).squeeze()


def norm_pct(pct: np.ndarray) -> np.ndarray:
    return np.divide(pct, 100)


def unnorm_pct(rate: np.ndarray) -> np.ndarray:
    return np.multiply(rate, 100)


setattr(sklearn.preprocessing, 'Clipper', partial(FunctionTransformer, func=np.clip, check_inverse=False))
setattr(sklearn.preprocessing, 'Identity', partial(FunctionTransformer, func=None, inverse_func=None, check_inverse=False))
setattr(sklearn.preprocessing, 'Log1pScaler', partial(FunctionTransformer, func=np.log1p, inverse_func=np.expm1, check_inverse=False))
setattr(sklearn.preprocessing, 'NaNAdder', partial(FunctionTransformer, func=partial(np.nansum, axis=1), check_inverse=False))
setattr(sklearn.preprocessing, 'PctNormalizer', partial(FunctionTransformer, func=norm_pct, inverse_func=unnorm_pct, check_inverse=False))
setattr(sklearn.preprocessing, 'Squeeze', partial(FunctionTransformer, func=squeeze_df, check_inverse=False))

try:  # only do this if dask_ml is installed
    from dask_ml.preprocessing import BlockTransformer
    import dask_ml.compose
    import dask_ml.feature_extraction
    import dask_ml.preprocessing
    # NOTE dask_ml BlockTransformer has no inverse_func
    setattr(dask_ml.preprocessing, 'Clipper', partial(BlockTransformer, func=np.clip))
    setattr(dask_ml.preprocessing, 'Identity', partial(BlockTransformer, func=ps.identity))
    setattr(dask_ml.preprocessing, 'Log1pScaler', partial(BlockTransformer, func=np.log1p))
    setattr(dask_ml.preprocessing, 'NaNAdder', partial(BlockTransformer, func=partial(np.nansum, axis=1), check_inverse=False))
    setattr(dask_ml.preprocessing, 'PctNormalizer', partial(BlockTransformer, func=norm_pct, check_inverse=False))
    setattr(dask_ml.preprocessing, 'Squeeze', partial(BlockTransformer, func=squeeze_df, check_inverse=False))
    # NOTE dask_ml OneHotEncoder is broken: https://github.com/dask/dask-ml/issues/548, https://github.com/dask/dask-ml/issues/623
    setattr(dask_ml.preprocessing, 'OneHotEncoder', sklearn.preprocessing.OneHotEncoder)
    setattr(dask_ml.feature_extraction, 'DictVectorizer', sklearn.feature_extraction.DictVectorizer)  # set missing class
except ImportError:
    pass


# transformed_names (feature names) methods


def get_ordinal_categories(spec: dict, pipeline: Pipeline) -> list:
    '''Get the first ordinal encoder categories from pipeline; used for embedding layer what consumes ordinal data'''
    for transformer in pipeline:
        if transformer.__class__.name == 'OrdinalEncoder':
            if ps.get(spec, 'dataset.transform.module') == 'dask_ml':
                return ps.values(transformer.dtypes_)[0].categories
            else:
                return transformer.categories_[0]


def _get_transformed_names(col_transfmr: ColumnTransformer, trans: Union[str, TransformerMixin], cols: list) -> list:
    if trans == 'drop' or (hasattr(cols, '__len__') and not len(cols)):
        return []
    if trans == 'passthrough':
        if hasattr(col_transfmr, '_df_columns'):
            if ((not isinstance(cols, slice)) and all(isinstance(col, str) for col in cols)):
                return cols
            else:
                return col_transfmr._df_columns[cols]
        else:
            indices = np.arange(col_transfmr._n_features)
            return [f'x{i}' for i in indices[cols]]

    if ps.is_list(cols):
        name = ps.join(cols, '_')
    else:
        name = cols
    # for transformers without get_feature_names_out method, use input names
    if not hasattr(trans, 'get_feature_names_out'):
        if cols is None:
            return []
        else:
            return [name]

    return [f'{col}' for col in trans.get_feature_names_out()]


def get_transformed_names(col_transfmr: ColumnTransformer) -> list:
    '''Get the transformed_names for ColumnTransformer'''
    # allow transformers to be pipelines. Pipeline steps are ordered differently, so need some processing
    if type(col_transfmr) == Pipeline:
        tfmrs_list = [(_auto_name, trans, None, None) for step, _auto_name, trans in col_transfmr._iter()]
    else:  # base case
        tfmrs_list = list(col_transfmr._iter(fitted=True))

    transformed_names = []
    for _auto_name, trans, cols, _ in tfmrs_list:
        if type(trans) == Pipeline:
            # Recursive call on pipeline
            names = get_transformed_names(trans)
            input_name = ps.join(cols, '_')
            if len(names) == 0:
                names = [input_name]
            else:
                names = [s.replace('None', input_name) for s in names]
            transformed_names.extend(names)
        else:
            transformed_names.extend(_get_transformed_names(col_transfmr, trans, cols))

    return transformed_names


# ColumnTransformer composer methods


def _get_module(module: str) -> type:
    if module == 'sklearn':
        return sklearn
    elif module == 'dask_ml':
        return dask_ml
    else:
        raise ValueError(f'Unrecognized module {module} not of sklearn, dask_ml')


def _get_transfmr_cls(name: str, module: str = 'sklearn') -> type:
    '''Get a transfmr class from sklearn/dask_ml(parallel) by scanning preprocessing then feature_extraction'''
    m = _get_module(module)
    try:
        Transfmr = getattr(m.preprocessing, name)
    except AttributeError:
        Transfmr = getattr(m.feature_extraction, name)
    return Transfmr


def _get_transfmr(name: str, module: str = 'sklearn', **kwargs) -> TransformerMixin:
    '''Get a transfmr from sklearn/dask_ml(parallel)'''
    Transfmr = _get_transfmr_cls(name, module)
    if isinstance(Transfmr, partial):  # handle custom transformers registered above
        if (t_func := Transfmr.func) == FunctionTransformer:
            return Transfmr(kw_args=kwargs)
        elif t_func == BlockTransformer:
            # Dask ColumnTransformer drops BlockTransformer.kw_args, so put it back via partial
            fn = Transfmr.keywords['func']
            return Transfmr(func=partial(fn, **kwargs))
        else:
            raise TypeError(f'Transfmr.func {t_func} is neither a sklearn.FunctionTransformer nor dask_ml.BlockTransformer')
    else:
        return Transfmr(**kwargs)


def _get_pipeline(trans: dict, module: str = 'sklearn') -> Pipeline:
    '''
    Get a pipeline of transfmrs for each entry in trans_spec, e.g.
    - {'Log1pScaler': None, 'StandardScaler': None}
    - {'Log1pScaler': None, 'Clipper': {'a_min': 0, 'a_max': 10}}
    '''
    assert ps.is_dict(trans), f'Transform spec must be a dict, but got {trans}'
    transfmrs = []
    for name, v in trans.items():
        kwargs = v or {}
        transfmrs.append(_get_transfmr(name, module, **kwargs))
    return make_pipeline(*transfmrs)


def get_col_transfmr(mode_trans_spec: dict, module: str = 'sklearn', n_jobs: int = -1) -> ColumnTransformer:
    '''
    Get a ColumnTransformer to transform dataframe into np matrix
    - mode_trans_spec format: {df_col: trans_spec}, where trans_spec is described in _get_pipeline()
    - module options: 'sklearn' (serial-row) or 'dask_ml' (parallel-row)
    - n_jobs: -1 to use all cores
    @examples

    mode_trans_spec = spec['transform'][mode]
    col_transfmr = get_col_transfmr(mode_trans_spec, module, n_jobs)
    if module == 'dask_ml':
        _fix_dask_df(df, spec)
    data = col_transfmr.fit_transform(df).astype(dtype)
    '''
    transformers = [
        # format: (name, pipeline, cols)
        (col, _get_pipeline(trans, module), [col])
        for col, trans in mode_trans_spec.items()
    ]
    m = _get_module(module)
    # ensure return of consistent np array and non-sparse matrix
    kwargs = {'perserve_dataframe': False, 'sparse_threshold': 0} if module == 'dask_ml' else {'sparse_threshold': 0}
    col_transfmr = m.compose.ColumnTransformer(transformers, n_jobs=n_jobs, **kwargs)
    return col_transfmr


def _fix_dask_df(df: pd.DataFrame, mode_trans_spec: dict) -> pd.DataFrame:
    '''Cast any columns for dask_ml OneHotEncoder and OrdinalEncoder to dtype 'category'. Apply this before fit_transform.'''
    fixed_cols = []
    for col, trans in mode_trans_spec.items():
        trans_names = [trans] if ps.is_string(trans) else list(trans)
        if any(trans_name in ['OneHotEncoder', 'OrdinalEncoder'] for trans_name in trans_names):
            df[col] = df[col].astype('category')
            fixed_cols.append(col)
    if len(fixed_cols):
        logger.info(f'Fixed cols {fixed_cols} dtype to "category" for dask_ml')


# ColumnTransformer interface methods


def get_filepath(spec: dict, mode: str, dir: Path = Path.cwd() / 'data', prefix='') -> str:
    sha_spec = ps.pick(spec, f'transform.{mode}')
    transform_sha = util.get_spec_sha(sha_spec)
    sub_names = [transform_sha, mode, 'col_transfmr.pkl']
    filename = ps.join(sub_names, '-')
    return dir / f'{prefix}{filename}'


def save_col_transfmr(col_transfmr: ColumnTransformer, spec: dict, mode: str) -> str:
    filepath = get_filepath(spec, mode)
    util.write(col_transfmr, filepath)
    logger.info(f'Saved col_transfmr to {filepath}')
    return filepath


def load_col_transfmr(spec: dict, mode: str) -> ColumnTransformer:
    filepath = get_filepath(spec, mode)
    col_transfmr = util.read(filepath)
    logger.info(f'Loaded col_transfmr from {filepath}')
    return col_transfmr


def get_fit_transform(df: pd.DataFrame, spec: dict, mode: str, dtype: type = np.float32, module: str = 'sklearn', n_jobs: int = -1, load: bool = False) -> tuple[np.ndarray, ColumnTransformer]:
    logger.info(f'Running for mode: {mode}')
    mode_trans_spec = spec['transform'][mode]
    if module == 'dask_ml':
        _fix_dask_df(df, mode_trans_spec)

    if load:
        col_transfmr = load_col_transfmr(spec, mode)
        trans_data = col_transfmr.transform(df).astype(dtype)
    else:
        col_transfmr = get_col_transfmr(mode_trans_spec, module, n_jobs)
        trans_data = col_transfmr.fit_transform(df).astype(dtype)
        setattr(col_transfmr, 'transformed_names_', get_transformed_names(col_transfmr))
        save_col_transfmr(col_transfmr, spec, mode)

    logger.info(f'{mode} shape {trans_data.shape}, features: {col_transfmr.transformed_names_}')
    logger.info('Done')
    return trans_data, col_transfmr


# transform module interface methods


def fit_transform(spec: dict, stage: str, df: pd.DataFrame) -> list:
    '''
    Fit transform input dataframe according to spec with the following format (in YAML):
    ```yaml
    dataset:
        transform:
            module: {str} # options: 'sklearn' (serial-row) or 'dask_ml' (parallel-row)
            n_jobs: {int} # parallelization; -1 to use all cores
    transform:
        {mode}:
            {column}:
                {preprocessor}: {null|kwargs}
                {preprocessor}: {null|kwargs}
                ...
    ```
    For example, the following spec (in YAML) transforms the iris dataframe by chaining Log1pScaler then StandardScalar to each column and produce the input `x`, and do one-hot encoding for the target column for output `y`:
    ```yaml
    dataset:
        transform:
            module: sklearn
            n_jobs: -1
    transform:
        x:
            sepal length (cm):
                Log1pScaler:
                StandardScaler:
            sepal width (cm):
                Log1pScaler:
                StandardScaler:
            petal length (cm):
                Log1pScaler:
                StandardScaler:
            petal width (cm):
                Log1pScaler:
                StandardScaler:
        y:
            target:
                OneHotEncoder:
                    sparse: false
                    handle_unknown: ignore
    ```
    @param spec:dict Specification of the transform
    @param stage:str Stage of the transform. If 'fit', create a new instance of ColumnTransformers and run fit_transform. If 'test' or 'validate', load existing ColumnTransformers and run transform.
    @param df:pd.DataFrame Input dataframe
    '''
    trans_data_list = []
    load = (stage != 'fit')  # other values used commonly in ml includes 'test', 'validate'. Those will load a fitted column transformer instead
    for mode in spec['transform']:
        trans_data, col_transfmr = get_fit_transform(df, spec, mode, dtype=np.float32, load=load, **spec['dataset']['transform'])
        trans_data_list.append(trans_data)
    return trans_data_list


def get_artifacts(spec: dict) -> dict:
    '''Get the artifacts for transformed data, including the col_transfmr and transformed_names'''
    artifacts = {
        'mode2col_transfmr': {},
        'mode2transformed_names': {},
    }
    try:
        for mode in spec['transform']:
            col_transfmr = load_col_transfmr(spec, mode)
            artifacts['mode2col_transfmr'][mode] = col_transfmr
            artifacts['mode2transformed_names'][mode] = col_transfmr.transformed_names_
        return artifacts
    except Exception as e:
        logger.exception('Could not find saved col_transfmrs to load artifacts from. This method needs to be called after data transform')
        raise e
