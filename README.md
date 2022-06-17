# feature_transform ![CI](https://github.com/kengz/feature_transform/workflows/CI/badge.svg)

Build ColumnTransformers ([Scikit](https://scikit-learn.org/stable/modules/generated/sklearn.compose.ColumnTransformer.html#sklearn.compose.ColumnTransformer) or [DaskML](https://ml.dask.org/modules/generated/dask_ml.compose.ColumnTransformer.html)) for feature transformation by specifying configs.

> For quickly building PyTorch models, see also [TorchArc](https://github.com/kengz/torcharc).

## Installation

```bash
pip install feature_transform
```

Installing this will also install Scikit Learn, but if you need parallelization, install Dask ML separately:

```bash
pip install dask-ml
```

## Usage

The ColumnTransformer class of [Scikit](https://scikit-learn.org/stable/modules/generated/sklearn.compose.ColumnTransformer.html#sklearn.compose.ColumnTransformer) / [DaskML](https://ml.dask.org/modules/generated/dask_ml.compose.ColumnTransformer.html) allows us to build a complex pipeline of feature preprocessors/transformers that takes dataframe as input and outputs numpy arrays. However, using it requires maintaining Python code.

This project started with the vision of building the entire feature transformation pipeline by just specifying what preprocessors to apply to a dataframe's column.

For example, take the iris dataset with columns: `sepal length (cm), sepal width (cm), petal length (cm), petal width (cm), target`. We want the first 4 columns to be the features for our input `x`, where each feature goes through a `StandardScaler`; and `target` to be the feature of our output `y`, where it is one-hot encoded. Then, use this directly to fit_transform the iris dataframe and obtain numpy arrays `xs, ys`. Here's the code:

```python
from feature_transform import transform
from sklearn import datasets
import pandas as pd


# specify transform for each feature
spec = {
    'dataset': {
        'transform': {'module': 'sklearn', 'n_jobs': 1}
    },
    'transform': {
        'x': { # the "mode"
            'sepal length (cm)': {'StandardScaler': None}, # the column name and its {preprocessor: kwargs, ...}
            'sepal width (cm)': {'StandardScaler': None},
            'petal length (cm)': {'StandardScaler': None},
            'petal width (cm)': {'StandardScaler': None},
        },
        'y': {
            'target': {'OneHotEncoder': {'sparse': False, 'handle_unknown': 'ignore'}}
        }
    }
}

# load iris dataframe
data_df = pd.concat(datasets.load_iris(return_X_y=True, as_frame=True), axis=1)
# transform into numpy arrays ready for model
mode2data = transform.fit_transform(spec, stage='fit', df=data_df)
xs, ys = mode2data['x'], mode2data['y']

# to reload the fitted transformers for validation/test, specify stage='validate' or 'test'
val_df = data_df.copy()
mode2val_data = transform.fit_transform(spec, stage='validate', df=val_df)
val_xs, val_ys = mode2val_data['x'], mode2val_data['y']

# artifacts to get the column transformers and transformed names directly
artifacts = transform.get_artifacts(spec)
artifacts['mode2col_transfmr']
# {'x': ColumnTransformer(n_jobs=1, sparse_threshold=0, transformers=[('sepal length (cm)', Pipeline(steps=[('standardscaler',...

artifacts['mode2transformed_names']
# {'x': ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)'],
#  'y': ['target_0', 'target_1', 'target_2']}
```

What happens in the background is as follows:

- for each `mode` in `spec.transform`
  - for each `column` in `mode`, create a pipeline of `[preprocessor(**kwargs)]`, and compose them into a `ColumnTransformer` for the mode.
  - during `fit_transform`, each mode runs its `ColumnTransformer.fit_transform`
  - then it saves the fitted `ColumnTransformer` to `./data/{hash}-{mode}-col_transfmr.pkl`.
  - these filenames will be logged. These files are the ones loaded in `transform.get_artifacts` for uses such as test/validation.

### Using YAML config

The goal of this library is to make feature transform configuration, so let's do the same as above, but with a YAML config file. The spec format is:

```yaml
dataset:
  transform:
    module: {str} # options: 'sklearn' (serial-row) or 'dask_ml' (parallel-row)
    n_jobs: {null|int} # parallelization; -1 to use all cores
transform:
  {mode}:
    {column}:
      {preprocessor}: {null|kwargs} # optional kwargs for preprocessor
      {preprocessor}: {null|kwargs}
      ...
```

The `{preprocessor}` value can be any of the preprocessor classes [Scikit](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.preprocessing) or [DaskML](https://ml.dask.org/modules/api.html#module-dask_ml.preprocessing). Additional custom ones are also registered in [feature_transform/transform.py](./feature_transform/transform.py).

For example, the earlier spec can be rewritten in YAML as:

```yaml
# transform.yaml
dataset:
  transform:
    module: sklearn
    n_jobs: null
transform:
  x:
    sepal length (cm):
      StandardScaler:
    sepal width (cm):
      StandardScaler:
    petal length (cm):
      StandardScaler:
    petal width (cm):
      StandardScaler:
  y:
    target:
      OneHotEncoder:
        sparse: false
        handle_unknown: ignore
```

Now, our code simplifies to:

```python
from feature_transform import transform, util
from sklearn import datasets
import pandas as pd


# convenient method to read YAML
spec = util.read('transform.yaml')
# load iris dataframe
data_df = pd.concat(datasets.load_iris(return_X_y=True, as_frame=True), axis=1)
# transform into numpy arrays ready for model
mode2data = transform.fit_transform(spec, stage='fit', df=data_df)
xs, ys = mode2data['x'], mode2data['y']

# to reload the fitted transformers for validation/test, specify stage='validate' or 'test'
val_df = data_df.copy()
mode2val_data = transform.fit_transform(spec, stage='validate', df=val_df)
val_xs, val_ys = mode2val_data['x'], mode2val_data['y']
```

### Chain Preprocessors

To chain multiple preprocessors, simply add more steps:

```yaml
dataset:
  transform:
    module: sklearn
    n_jobs: null
transform:
  x:
    sepal length (cm):
      Log1pScaler: # custom preprocessor for np.log1p
      StandardScaler:
    sepal width (cm):
      Clipper: # custom preprocessor to clip values
        a_min: 0
        a_max: 10
      StandardScaler:
    petal length (cm):
      StandardScaler:
    petal width (cm):
      StandardScaler:
  y:
    target:
      OneHotEncoder:
        sparse: false
        handle_unknown: ignore
```

### Specify any modes

The modes can be any names other than `x, y`:

```yaml
dataset:
  transform:
    module: sklearn
    n_jobs: null
transform:
  foo:
    column_foo_1:
      StandardScaler:
    column_foo_2:
      Log1pScaler:
      StandardScaler:
  bar:
    column_bar_1:
      OneHotEncoder:
  baz:
    column_baz_1:
      Identity:
```

### Parallelization

> NOTE run `pip install dask-ml` first.

```yaml
dataset:
  transform:
    module: dask_ml
    n_jobs: -1 # use all cores
transform:
  # ...
```

## ML Examples

### PyTorch DataLoader

```python
from feature_transform import transform, util
from sklearn import datasets
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import torch


spec = util.read('transform.yaml')
# load iris dataframe
data_df = pd.concat(datasets.load_iris(return_X_y=True, as_frame=True), axis=1)
# transform into numpy arrays ready for model
mode2data = transform.fit_transform(spec, stage='fit', df=data_df)
xs, ys = mode2data['x'], mode2data['y']

train_dataset = TensorDataset(torch.from_numpy(xs), torch.from_numpy(ys)) # create your datset
train_dataloader = DataLoader(train_dataset) # create your dataloader

# suppose this is test/validation set; use stage='validate' or stage='test' to transform
val_df = data_df.copy()
mode2val_data = transform.fit_transform(spec, stage='validate', df=val_df)
val_xs, val_ys = mode2val_data['x'], mode2val_data['y']
val_dataset = TensorDataset(torch.from_numpy(val_xs), torch.from_numpy(val_ys))
val_dataloader = DataLoader(val_dataset) # create your dataloader
```

### Scikit Learn example

```python
from feature_transform import transform, util
from sklearn import datasets, metrics
from sklearn.tree import DecisionTreeClassifier
import pandas as pd


spec = util.read('transform.yaml')
# load iris dataframe
data_df = pd.concat(datasets.load_iris(return_X_y=True, as_frame=True), axis=1)
# transform into numpy arrays ready for model
mode2data = transform.fit_transform(spec, stage='fit', df=data_df)
xs, ys = mode2data['x'], mode2data['y']

# train model
model = DecisionTreeClassifier(max_depth = 3, random_state = 1)
model.fit(xs, ys)
pred_ys = model.predict(xs)
print(f'train accuracy: {metrics.accuracy_score(pred_ys, ys):.3f}')
# train accuracy: 0.973

# suppose this is validation/test data, we use stage='validate' or 'test
test_df = data_df.copy()
mode2test_data = transform.fit_transform(spec, stage='test', df=test_df)
test_xs, test_ys = mode2val_data['x'], mode2val_data['y']
pred_ys = model.predict(test_xs)
print(f'test accuracy: {metrics.accuracy_score(pred_ys, test_ys):.3f}')
# test accuracy: 0.973
```

## Development

### Setup

```bash
# install the dev dependencies
bin/setup
# activate Conda environment
conda activate transform
```

### Unit Tests

```bash
python setup.py test
```
