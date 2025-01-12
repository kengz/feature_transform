from collections import defaultdict

import numpy as np
import pandas as pd
import polars as pl
from scipy.stats import shapiro


def suggest_sr(
    series: pd.Series | pl.Series,
    skew_threshold: float = 1.0,
    p_value_threshold: float = 0.05,
) -> list[str]:
    """
    Suggest sklearn data transformer (preprocessing) steps for a given Series (pandas or polars).

    Parameters:
        series (pd.Series or pl.Series): The input Series.
        skew_threshold (float): The skewness threshold to apply log transformation.
        p_value_threshold (float): The p-value threshold for normality testing.

    Returns:
        list[str]: A list of sklearn preprocessing steps.
    """
    suggestions = []

    # Determine if input is pandas or polars
    is_pandas = isinstance(series, pd.Series)
    is_polars = isinstance(series, pl.Series)
    if not (is_pandas or is_polars):
        raise TypeError("Input must be a pandas.Series or polars.Series.")

    # Handle missing values
    if is_pandas and series.isnull().any():
        suggestions.append("impute.SimpleImputer")
    elif is_polars and series.is_null().any():
        suggestions.append("impute.SimpleImputer")

    # Handle categorical data
    if is_pandas and (
        series.dtype == "object" or str(series.dtype).startswith("category")
    ):
        suggestions.append("preprocessing.OneHotEncoder")
        return suggestions
    elif is_polars and series.dtype in [pl.Utf8, pl.Categorical]:
        suggestions.append("preprocessing.OneHotEncoder")
        return suggestions

    # Check if numeric
    if (is_pandas and np.issubdtype(series.dtype, np.number)) or (
        is_polars and series.dtype in [pl.Float64, pl.Int64]
    ):
        # Drop nulls for processing
        data = series.dropna() if is_pandas else series.drop_nulls()

        # Skewness check
        skewness = data.skew() if is_pandas else data.skew()
        if abs(skewness) > skew_threshold:
            suggestions.append("preprocessing.PowerTransformer")
            return suggestions

        # Normality check
        data_values = data.values if is_pandas else data.to_numpy()
        shapiro_stat, p_value = shapiro(data_values)
        is_normal = p_value > p_value_threshold

        # Outlier check using IQR
        q1 = data.quantile(0.25) if is_pandas else data.quantile(0.25)
        q3 = data.quantile(0.75) if is_pandas else data.quantile(0.75)
        iqr = q3 - q1
        outlier_threshold = 1.5 * iqr
        has_outliers = (
            (data < (q1 - outlier_threshold)) | (data > (q3 + outlier_threshold))
        ).any()

        # Scaling suggestions
        if is_normal and not has_outliers:
            suggestions.append("preprocessing.StandardScaler")
        else:
            suggestions.append("preprocessing.RobustScaler")

    return suggestions


def suggest(
    df: pd.DataFrame | pl.DataFrame,
    skew_threshold: float = 1.0,
    p_value_threshold: float = 0.05,
) -> dict:
    """
    Suggest Feature Transform spec for a dataframe by suggesting sklearn data transformer (preprocessing) steps for each of its columns.
    Returns Feature Transform spec_dict for inspection/editing/use in ft.build(spec_dict) to build ColumnTransformer

    Parameters:
        df (pd.DataFrame or pl.DataFrame): The input DataFrame.
        skew_threshold (float): The skewness threshold to apply log transformation.
        p_value_threshold (float): The p-value threshold for normality testing.

    Returns:
        spec_dict (dict): for ft.build(spec_dict) to build ColumnTransformer
    """
    # suggest for each column
    col2cls_names = {
        col: suggest_sr(
            df[col], skew_threshold=skew_threshold, p_value_threshold=p_value_threshold
        )
        for col in df.columns
    }
    # Flip the dictionary to group columns by the same cls_names
    cls_names2cols = defaultdict(list)
    for col, cls_names in col2cls_names.items():
        cls_names2cols[tuple(cls_names)].append(col)
    cls_names2cols = dict(cls_names2cols)

    # build transformers spec
    transformers = []
    for cls_names, columns in cls_names2cols.items():
        if len(cls_names) > 1:  # pipeline
            pipeline = [{name: {}} for name in cls_names]
            tfm_spec = {"transformer": {"Pipeline": pipeline}, "columns": columns}
        else:  # single processor
            name = cls_names[0]
            tfm_spec = {"transformer": {name: {}}, "columns": columns}
        transformers.append(tfm_spec)
    return {"transformers": transformers}
