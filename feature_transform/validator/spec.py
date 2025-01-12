import operator
from pathlib import Path

import sklearn
import yaml
from pydantic import BaseModel, Field, RootModel, field_validator


class EstimatorSpec(RootModel):
    """
    Basic spec to construct a sklearn class, e.g. Estimator (with fit and transform), where key = class name, value = kwargs, i.e. sklearn.<key>(**value).
    Must be a single-key dict.
    E.g.
    preprocessing.OneHotEncoder:
        sparse_output: False
    """

    root: dict[str, dict | None] = Field(
        description="Spec for sklearn class, where key = class name, value = kwargs.",
        examples=[
            {"preprocessing.StandardScaler": {}},
            {"preprocessing.StandardScaler": None},
            {
                "preprocessing.OneHotEncoder": {
                    "sparse_output": False,
                }
            },
        ],
    )

    @field_validator("root", mode="before")
    def is_single_key_dict(value: dict) -> dict:
        assert len(value) == 1, "Class spec must be a single-key dict."
        return value

    @field_validator("root", mode="before")
    def null_kwargs_to_dict(value: dict) -> dict:
        key = next(iter(value))
        if value[key] is None:
            value[key] = {}
        return value

    @field_validator("root", mode="after")
    def key_exists_in_sklearn(value: dict) -> dict:
        class_path = next(iter(value))
        assert operator.attrgetter(class_path)(sklearn) is not None
        return value

    def build(self):
        """Build sklearn class from spec."""
        class_path = next(iter(self.root))
        cls = operator.attrgetter(class_path)(sklearn)
        kwargs = self.root[class_path]
        return cls(**kwargs)


class PipelineSpec(BaseModel):
    """
    Spec for Pipeline=List[EstimatorSpec] to build a Pipeline using `make_pipeline`, which is a shorthand that doesn't require names - estimator names will be set to the lowercase of their types automatically.
    See https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.make_pipeline.html
    E.g.
    Pipeline:
        - impute.SimpleImputer:
            strategy: constant
        - preprocessing.StandardScaler:
    """

    Pipeline: list[EstimatorSpec] = Field(
        description="List of Estimator that are chained together. See https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.make_pipeline.html",
        examples=[
            {
                "Pipeline": [
                    {
                        "impute.SimpleImputer": {
                            "strategy": "constant",
                        },
                    },
                    {"preprocessing.StandardScaler": {}},
                ]
            }
        ],
    )

    def build(self) -> sklearn.pipeline.Pipeline:
        return sklearn.pipeline.make_pipeline(
            *[tuple_spec.build() for tuple_spec in self.Pipeline]
        )


class TransformerSpec(BaseModel):
    """
    Spec for either (name, transformer, columns) tuples for ColumnTransformer https://scikit-learn.org/stable/modules/generated/sklearn.compose.ColumnTransformer.html
    or (transformer, columns) tuples for make_column_transformer https://scikit-learn.org/stable/modules/generated/sklearn.compose.make_column_transformer.html
    E.g.
    transformer:
        preprocessing.StandardScaler:
    columns: [col0, col1]

    E.g.
    name: std
    transformer:
        Pipeline:
            - impute.SimpleImputer:
                strategy: constant
            - preprocessing.StandardScaler:
    columns: [col0, col1]
    """

    name: str | None = Field(
        None,
        description="Mame of the transformer - include to use ColumnTransformer; exclude to use make_column_transformer",
        examples=["std"],
    )
    transformer: str | EstimatorSpec | PipelineSpec = Field(
        description="{'drop', 'passthrough'} or estimator. Estimator must support fit and transform. Special-cased strings 'drop' and 'passthrough' are accepted as well, to indicate to drop the columns or to pass them through untransformed, respectively.",
        examples=[
            "drop",
            {"preprocessing.StandardScaler": {}},
            {
                "preprocessing.OneHotEncoder": {
                    "sparse_output": False,
                }
            },
            {
                "Pipeline": [
                    {
                        "impute.SimpleImputer": {
                            "strategy": "constant",
                        },
                    },
                    {"preprocessing.StandardScaler": {}},
                ]
            },
        ],
    )
    columns: str | int | list[str | int] = Field(
        description="Indexes the data on its second axis. Integers are interpreted as positional columns, while strings can reference DataFrame columns by name.",
        examples=[
            "col1",
            ["col1", "col2"],
            0,
            [0, 1],
        ],
    )

    def build(self) -> tuple:
        if isinstance(self.transformer, str):
            transformer = self.transformer
        else:
            transformer = self.transformer.build()
        if self.name:
            return (self.name, transformer, self.columns)
        else:
            return (transformer, self.columns)


class Spec(BaseModel):
    """
    Main spec to create ColumnTransformer.
    If transformers=list[(name, transformer, columns)], use ColumnTransformer https://scikit-learn.org/stable/modules/generated/sklearn.compose.ColumnTransformer.html
    If transformers=list[(transformer, columns)] use make_column_transformer https://scikit-learn.org/stable/modules/generated/sklearn.compose.make_column_transformer.html
    E.g.
    transformers:
        - name: std
          transformer:
            preprocessing.StandardScaler:
          columns: [col1, col2]
        - name: impute_std
          transformer:
            Pipeline:
                - impute.SimpleImputer:
                    strategy: constant
                - preprocessing.StandardScaler:
          columns: [col3, col4]
    """

    transformers: list[TransformerSpec] = Field(
        description="List of (name, transformer, columns) for ColumnTransformer or List of (transformer, columns) for make_column_transformer where name is given automatically by estimator type.",
        examples=[
            # basic
            {
                "transformers": [
                    {
                        "transformer": {"preprocessing.StandardScaler": {}},
                        "columns": ["col1", "col2"],
                    },
                    {
                        "transformer": {"preprocessing.RobustScaler": {}},
                        "columns": ["col3", "col4"],
                    },
                ]
            },
            # name; int-columns
            {
                "transformers": [
                    {
                        "name": "std",
                        "transformer": {"preprocessing.StandardScaler": {}},
                        "columns": [0, 1],
                    }
                ]
            },
            # pipeline
            {
                "transformers": [
                    {
                        "transformer": {"preprocessing.StandardScaler": {}},
                        "columns": ["col1", "col2"],
                    },
                    {
                        "transformer": {
                            "Pipeline": [
                                {
                                    "impute.SimpleImputer": {
                                        "strategy": "constant",
                                    },
                                },
                                {
                                    "preprocessing.OneHotEncoder": {
                                        "sparse_output": False,
                                    }
                                },
                            ]
                        },
                        "columns": ["col3", "col4"],
                    },
                ]
            },
            # ColumnTransformer settings
            {
                "transformers": [
                    {
                        "transformer": {"preprocessing.StandardScaler": {}},
                        "columns": ["col1", "col2"],
                    }
                ],
                # use all processors
                "n_jobs": -1,
            },
        ],
    )
    # other settings following the defaults for ColumnTransformer / make_column_transformer
    remainder: str = "drop"
    sparse_threshold: float = 0.3
    n_jobs: int | None = None
    transformer_weights: dict | None = None
    verbose: bool = False
    verbose_feature_names_out: bool = True
    force_int_remainder_cols: bool = True

    @field_validator("transformers", mode="before")
    def check_tuple_same_size(value: list):
        sizes = [len(t_spec) for t_spec in value]
        assert (
            len(set(sizes)) == 1
        ), "All TransformerSpecs must uniformly specify name or not."
        return value

    def build(self) -> sklearn.compose.ColumnTransformer:
        transformers = [t.build() for t in self.transformers]
        if len(transformers[0]) == 3:
            return sklearn.compose.ColumnTransformer(
                transformers,
                remainder=self.remainder,
                sparse_threshold=self.sparse_threshold,
                n_jobs=self.n_jobs,
                transformer_weights=self.transformer_weights,
                verbose=self.verbose,
                verbose_feature_names_out=self.verbose_feature_names_out,
                force_int_remainder_cols=self.force_int_remainder_cols,
            )
        else:
            return sklearn.compose.make_column_transformer(
                *transformers,
                remainder=self.remainder,
                sparse_threshold=self.sparse_threshold,
                n_jobs=self.n_jobs,
                verbose=self.verbose,
                verbose_feature_names_out=self.verbose_feature_names_out,
                force_int_remainder_cols=self.force_int_remainder_cols,
            )


def build(spec: dict | str) -> sklearn.compose.ColumnTransformer:
    """
    Build a ColumnTransformer from a spec dict or path to a spec dict, e.g.

    import yaml

    import feature_transform as ft

    yaml_str = '''
    transformers:
      - transformer:
          preprocessing.StandardScaler:
        columns: [alcohol, total_phenols]
      - transformer:
          preprocessing.RobustScaler:
        columns: [ash]
    '''
    spec = yaml.safe_load(yaml_str)

    col_tfm = ft.build(spec)
    print(col_tfm)
    # ColumnTransformer(
    #     transformers=[
    #         ("standardscaler", StandardScaler(), ["alcohol", "total_phenols"]),
    #         ("robustscaler", RobustScaler(), ["ash"]),
    #     ]
    # )
    """
    if not isinstance(spec, dict):
        with Path(spec).open("r") as f:
            spec = yaml.safe_load(f)
    return Spec(**spec).build()
