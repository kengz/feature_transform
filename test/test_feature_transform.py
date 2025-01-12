import pytest
import sklearn
from sklearn.base import BaseEstimator, TransformerMixin

import feature_transform as ft

spec_dict = {
    "transformers": [
        {
            "transformer": {"preprocessing.StandardScaler": {}},
            "columns": ["alcohol", "total_phenols"],
        },
        {
            "transformer": {"preprocessing.RobustScaler": {}},
            "columns": ["ash"],
        },
    ]
}


def test_spec():
    col_tfm = ft.Spec(**spec_dict).build()
    assert isinstance(col_tfm, sklearn.compose.ColumnTransformer)


def test_build_from_file():
    spec_file = ft.SPEC_DIR / "basic.yaml"
    col_tfm = ft.build(spec_file)
    assert isinstance(col_tfm, sklearn.compose.ColumnTransformer)


def test_build_from_dict():
    col_tfm = ft.build(spec_dict)
    assert isinstance(col_tfm, sklearn.compose.ColumnTransformer)


def test_register_class():
    class Dummy(BaseEstimator, TransformerMixin):
        def __init__(self):
            pass

        def fit(self, X, y=None):
            return self

        def transform(self, X):
            return X

    ft.register_class(Dummy)

    # conflict with existing class - not allowed
    with pytest.raises(ValueError):
        ft.register_class(Dummy)


# test all the example specs
@pytest.mark.parametrize("spec_file", list(ft.SPEC_DIR.rglob("*.yaml")))
def test_build(spec_file):
    # Build ColumnTransformer using ft
    col_tfm = ft.build(spec_file)
    assert isinstance(col_tfm, sklearn.compose.ColumnTransformer)
