# Feature Transform

![Test](https://github.com/github/docs/actions/workflows/test.yml/badge.svg)

Build [Scikit ColumnTransformers](https://scikit-learn.org/stable/modules/generated/sklearn.compose.ColumnTransformer.html) by specifying configs.

> See also [TorchArc](https://github.com/kengz/torcharc) to build PyTorch models by specifying architectures.

## Installation

```bash
pip install feature_transform
```

## Usage

1. specify column transformers in a YAML spec file, e.g. at `spec_filepath = "./example/spec/basic.yaml"`
2. `import feature_transform as ft`.
   1. (optional) if you have custom sklearn estimator/preprocessor, e.g. `Dummy`, register it with `ft.register_class(Dummy)`
3. build with: `col_tfm = ft.build(spec_filepath)`

The returned object is a sklearn `ColumnTransformer` ready for normal use.

See more examples below, then see how it works at the end.

---

### Example: build ColumnTransformer from spec file

```python
from pathlib import Path

import joblib
import yaml
from sklearn import datasets

import feature_transform as ft

filepath = Path(".") / "feature_transform" / "example" / "spec" / "basic.yaml"

# The following are equivalent:

# 1. build from YAML spec file
col_tfm = ft.build(filepath)

# 2. build from dictionary
with filepath.open("r") as f:
    spec_dict = yaml.safe_load(f)
col_tfm = ft.build(spec_dict)

# 3. use the underlying Pydantic validator to build the col_tfm
spec = ft.Spec(**spec_dict)
col_tfm = spec.build()
```

---

Next, load demo data for examples below:

```python
# ================================================
# Load demo data

x_df, y_sr = datasets.load_wine(return_X_y=True, as_frame=True)

x_df.columns
# Index(['alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash', 'magnesium',
#        'total_phenols', 'flavanoids', 'nonflavanoid_phenols',
#        'proanthocyanins', 'color_intensity', 'hue',
#        'od280/od315_of_diluted_wines', 'proline'],
#       dtype='object')
```

---

### Example: basic

Spec file: [feature_transform/example/spec/basic.yaml](feature_transform/example/spec/basic.yaml)

```yaml
transformers:
  - transformer:
      preprocessing.StandardScaler:
    columns: [alcohol, total_phenols]
  - transformer:
      preprocessing.RobustScaler:
    columns: [ash]
```

```python
col_tfm = ft.build(ft.SPEC_DIR / "basic.yaml")

feat_xs = col_tfm.fit_transform(x_df)
feat_xs
# array([[ 1.51861254,  0.80899739,  0.20143885],
#        ...,

# save for later use
joblib.dump(col_tfm, "col_tfm.joblib")

# ... later, e.g. during batch inference
loaded_col_tfm = joblib.load("col_tfm.joblib")
feat_xs = loaded_col_tfm.transform(x_df)
```

ColumnTransformer `col_tfm`:

![](images/basic.png)

---

### Example: basic with pandas/polars dataframe

Spec file: [feature_transform/example/spec/basic.yaml](feature_transform/example/spec/basic.yaml)

```yaml
transformers:
  - transformer:
      preprocessing.StandardScaler:
    columns: [alcohol, total_phenols]
  - transformer:
      preprocessing.RobustScaler:
    columns: [ash]
```

```python
col_tfm = ft.build(ft.SPEC_DIR / "basic.yaml")
# to use with dataframe, set output to "pandas" or "polars"
col_tfm.set_output(transform="pandas")

feat_x_df = col_tfm.fit_transform(x_df)
feat_x_df
# 	standardscaler__alcohol	standardscaler__total_phenols	robustscaler__ash
# 0	1.518613	0.808997	0.201439
# 1	0.246290	0.568648	-0.633094
# ...

feat_x_df.describe()
# 	standardscaler__alcohol	standardscaler__total_phenols	robustscaler__ash
# count	1.780000e+02	178.000000	178.000000
# mean	-8.382808e-16	0.000000	0.018754
# std	1.002821e+00	1.002821	0.789479
# ...

# save for later use
joblib.dump(col_tfm, "col_tfm.joblib")

# ... later, e.g. during batch inference
loaded_col_tfm = joblib.load("col_tfm.joblib")
feat_x_df = loaded_col_tfm.transform(x_df)
```

ColumnTransformer `col_tfm`:

![](images/basic.png)

---

### Example: specify name; use int columns

Spec file: [feature_transform/example/spec/name-intcol.yaml](feature_transform/example/spec/name-intcol.yaml)

```yaml
transformers:
  - name: std
    transformer:
      preprocessing.StandardScaler:
    columns: [0, 5]
  - name: robust
    transformer:
      preprocessing.RobustScaler:
    columns: [2]
```

```python
col_tfm = ft.build(ft.SPEC_DIR / "name-intcol.yaml")

feat_xs = col_tfm.fit_transform(x_df)
# array([[ 1.51861254,  0.80899739,  0.20143885],
#        ...,
```

ColumnTransformer `col_tfm`:

![](images/name-intcol.png)

---

### Example: pipeline

Spec file: [feature_transform/example/spec/pipeline.yaml](feature_transform/example/spec/pipeline.yaml)

```yaml
transformers:
  - transformer:
      preprocessing.StandardScaler:
    columns: [alcohol, total_phenols]
  - transformer:
      Pipeline:
        - impute.SimpleImputer:
            strategy: constant
        - preprocessing.RobustScaler:
    columns: [ash]
```

```python
col_tfm = ft.build(ft.SPEC_DIR / "pipeline.yaml")

feat_xs = col_tfm.fit_transform(x_df)
feat_xs
# array([[ 1.51861254,  0.80899739,  0.20143885],
#        ...,
```

ColumnTransformer `col_tfm`:

![](images/pipeline.png)

---

### Example: ColumnTransformer settings

Spec file: [feature_transform/example/spec/settings.yaml](feature_transform/example/spec/settings.yaml)

```yaml
transformers:
  - transformer:
      preprocessing.StandardScaler:
    columns: [alcohol, total_phenols]
  - transformer:
      preprocessing.RobustScaler:
    columns: [ash]
# use all processors
n_jobs: -1
# for more kwargs see https://scikit-learn.org/stable/modules/generated/sklearn.compose.make_column_transformer.html
```

```python
col_tfm = ft.build(ft.SPEC_DIR / "settings.yaml")

feat_xs = col_tfm.fit_transform(x_df)
feat_xs
# array([[ 1.51861254,  0.80899739,  0.20143885],
#        ...,
```

ColumnTransformer `col_tfm`:

![](images/settings.png)

---

### Example: full X, y feature transform with save/load

Spec file (x): [feature_transform/example/spec/wine/x.yaml](feature_transform/example/spec/wine/x.yaml)

```yaml
transformers:
  - transformer:
      preprocessing.StandardScaler:
    columns: [alcohol, total_phenols, flavanoids, nonflavanoid_phenols, od280/od315_of_diluted_wines]
  - transformer:
      preprocessing.RobustScaler:
    columns: [ash, alcalinity_of_ash, proanthocyanins, hue]
  - transformer:
      preprocessing.PowerTransformer:
    columns: [malic_acid, magnesium, color_intensity, proline]
n_jobs: -1
```

Spec file (y): [feature_transform/example/spec/wine/y.yaml](feature_transform/example/spec/wine/y.yaml)

```yaml
transformers:
  - transformer:
      preprocessing.OneHotEncoder:
        sparse_output: False
    columns: [target]
```

```python
import joblib
from sklearn import datasets

import feature_transform as ft

x_df, y_sr = datasets.load_wine(return_X_y=True, as_frame=True)
y_df = y_sr.to_frame()  # ColumnTransformer takes only dataframe/matrix as input

x_col_tfm = ft.build(ft.SPEC_DIR / "wine" / "x.yaml")
y_col_tfm = ft.build(ft.SPEC_DIR / "wine" / "y.yaml")

# fit-transform
feat_xs = x_col_tfm.fit_transform(x_df)
feat_xs
# array([[ 1.51861254,  0.80899739,  1.03481896, ...,  1.69074868,
#          0.45145022,  1.06254129],
#        ...,

feat_ys = y_col_tfm.fit_transform(y_df)
feat_ys
# array([[1., 0., 0.],
#        ...,

# save for later use
joblib.dump(x_col_tfm, "x_col_tfm.joblib")
joblib.dump(y_col_tfm, "y_col_tfm.joblib")


# ... later, e.g. during batch inference
loaded_x_col_tfm = joblib.load("x_col_tfm.joblib")
feat_xs = loaded_x_col_tfm.transform(x_df)
feat_xs
# array([[ 1.51861254,  0.80899739,  1.03481896, ...,  1.69074868,
#          0.45145022,  1.06254129],
#        ...,
```

ColumnTransformer `x_col_tfm`:

![](images/wine-x.png)

ColumnTransformer `y_col_tfm`:

![](images/wine-y.png)

---

### Example: use helper to suggest spec

Most of the time, data preprocessing steps can be determined with rules-of-thumb; `ft.suggest` does exactly that (see [feature_transform/helper.py](feature_transform/helper.py) for details). This produces `spec_dict` that can be used directly with `ft.build` or for further editing.

```python
x_df, y_sr = datasets.load_wine(return_X_y=True, as_frame=True)

# suggest spec_dict - use directly or save to yaml for further editing
spec_dict = ft.suggest(x_df)
col_tfm = ft.build(spec_dict)

# fit-transform
feat_xs = col_tfm.fit_transform(x_df)
feat_xs
# array([[ 0.8973384 ,  0.20143885, -0.90697674, ...,  0.80804954,
#         -0.43546273,  1.69074868],
#         ...,
```

ColumnTransformer `col_tfm`:

![](images/suggest.png)

---

### Example: more

See more examples:

- demo notebook from above [feature_transform/example/notebook/demo.py](feature_transform/example/notebook/demo.py)
- spec files [feature_transform/example/spec/](feature_transform/example/spec/)
- unit tests [test/validator/test_spec.py](test/validator/test_spec.py)

## How does it work

Feature Transform simply builds sklearn ColumnTransformer and its estimators/pipelines with 1-1 mapping from a spec file:

1. Spec is defined via Pydantic [feature_transform/validator/](feature_transform/validator/). This defines:
   - `spec`: the `Estimator, Pipeline, ColumnTransformer`
2. If spec specifies:
   1. `transformers=list[(name, transformer, columns)]`, then use [`ColumnTransformer`](https://scikit-learn.org/stable/modules/generated/sklearn.compose.ColumnTransformer.html)
   1. `transformers=list[(transformer, columns)]`, then use [`make_column_transformer`](https://scikit-learn.org/stable/modules/generated/sklearn.compose.make_column_transformer.html) with auto-generated names

See more in the pydantic spec definition:

- [feature_transform/validator/spec.py](feature_transform/validator/spec.py): the spec used by feature_transform

### Guiding principles

The design of Feature Transform is guided as follows:

1. simple: the module spec is straightforward:
   1. it is simply sklearn class name with kwargs.
   1. it supports official `sklearn` estimators, `Pipeline`, and custom-defined modules registered via `ft.register_class`
1. expressive: it can be used to build both simple and advanced `ColumnTransformer` easily
1. portable: it returns `ColumnTransformer` that can be used anywhere; it is not a framework.
1. parametrizable: data-based feature transformation unlocks fast experimentation, e.g. by building logic for hyperparameter / data feature search

## Development

### Setup

[Install uv](https://docs.astral.sh/uv/getting-started/installation/) for dependency management if you haven't already. Then run:

```bash
# setup virtualenv
uv sync
```

### Unit Tests

```bash
uv run pytest
```
