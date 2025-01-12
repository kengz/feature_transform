import pytest
import sklearn

from feature_transform.validator.spec import (
    EstimatorSpec,
    PipelineSpec,
    Spec,
    TransformerSpec,
)


@pytest.mark.parametrize(
    "spec_dict",
    [
        {"preprocessing.StandardScaler": {}},
        {"preprocessing.StandardScaler": None},
        {
            "preprocessing.OneHotEncoder": {
                "sparse_output": False,
            }
        },
    ],
)
def test_estimator_spec(spec_dict):
    estimator = EstimatorSpec(**spec_dict).build()
    assert hasattr(estimator, "fit")
    assert hasattr(estimator, "transform")


@pytest.mark.parametrize(
    "spec_dict",
    [
        # multi-key
        {"preprocessing.StandardScaler": {}, "preprocessing.RobustScaler": {}},
        # invalid/unregistered sklearn path
        {"Invalid": {}},
        # key must be sklearn.<key>
        {"StandardScaler": {}},
        # init error
        {"preprocessing.StandardScaler": {"invalid": 64}},
    ],
)
def test_invalid_estimator_spec(spec_dict):
    with pytest.raises(Exception):
        EstimatorSpec(**spec_dict).build()


@pytest.mark.parametrize(
    "spec_dict",
    [
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
def test_pipeline_spec(spec_dict):
    pipeline = PipelineSpec(**spec_dict).build()
    assert isinstance(pipeline, sklearn.pipeline.Pipeline)
    assert hasattr(pipeline, "transform")


@pytest.mark.parametrize(
    "spec_dict",
    [
        # non-list
        {"Sequential": {"preprocessing.StandardScaler": {}}},
        # non-Pipeline
        {"StandardScaler": {}},
    ],
)
def test_invalid_pipeline_spec(spec_dict):
    with pytest.raises(Exception):
        PipelineSpec(**spec_dict).build()


@pytest.mark.parametrize(
    "spec_dict",
    [
        # 'drop' or 'passthrough' str
        {"name": "pass", "transformer": "passthrough", "columns": ["col1", "col2"]},
        # basic transformer
        {
            "transformer": {"preprocessing.StandardScaler": {}},
            "columns": ["col1", "col2"],
        },
        # pipeline transformer
        {
            "transformer": {
                "Pipeline": [
                    {
                        "impute.SimpleImputer": {
                            "strategy": "constant",
                        },
                    },
                    {"preprocessing.StandardScaler": {}},
                ]
            },
            "columns": ["col1", "col2"],
        },
        # with name (auto-generated)
        {
            "name": "std",
            "transformer": {"preprocessing.StandardScaler": {}},
            "columns": ["col1", "col2"],
        },
        # int columns
        {
            "transformer": {"preprocessing.StandardScaler": {}},
            "columns": [0, 1],
        },
    ],
)
def test_transformer_spec(spec_dict):
    transformer_tuple = TransformerSpec(**spec_dict).build()
    assert len(transformer_tuple) == len(spec_dict)


@pytest.mark.parametrize(
    "spec_dict",
    [
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
def test_column_transformer_spec(spec_dict):
    col_tfm = Spec(**spec_dict).build()
    assert isinstance(col_tfm, sklearn.compose.ColumnTransformer)


@pytest.mark.parametrize(
    "spec_dict",
    [
        # multiple but non-uniform spec of "names"
        {
            "transformers": [
                {
                    "names": "std",
                    "transformer": {"preprocessing.StandardScaler": {}},
                    "columns": ["col1", "col2"],
                },
                {
                    "transformer": {"preprocessing.RobustScaler": {}},
                    "columns": ["col3"],
                },
            ]
        },
    ],
)
def test_invalid_column_transformer_spec(spec_dict):
    with pytest.raises(Exception):
        Spec(**spec_dict).build()
