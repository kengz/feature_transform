from pathlib import Path

import joblib
import yaml
from sklearn import datasets

import feature_transform as ft

# ================================================
# Example: build col_tfm from spec file
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


# ================================================
# Load demo data

x_df, y_sr = datasets.load_wine(return_X_y=True, as_frame=True)

x_df.columns
# Index(['alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash', 'magnesium',
#        'total_phenols', 'flavanoids', 'nonflavanoid_phenols',
#        'proanthocyanins', 'color_intensity', 'hue',
#        'od280/od315_of_diluted_wines', 'proline'],
#       dtype='object')


# ================================================
# Example: basic
col_tfm = ft.build(ft.SPEC_DIR / "basic.yaml")
col_tfm
# ColumnTransformer(
#     transformers=[
#         ("standardscaler", StandardScaler(), ["alcohol", "total_phenols"]),
#         ("robustscaler", RobustScaler(), ["ash"]),
#     ]
# )

feat_xs = col_tfm.fit_transform(x_df)
feat_xs
# array([[ 1.51861254,  0.80899739,  0.20143885],
#        ...,

# save for later use
joblib.dump(col_tfm, "col_tfm.joblib")

# ... later, e.g. during batch inference
loaded_col_tfm = joblib.load("col_tfm.joblib")
feat_xs = loaded_col_tfm.transform(x_df)


# ================================================
# Example: basic with pandas/polars dataframe
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


# ================================================
# Example: specify name; use int columns
col_tfm = ft.build(ft.SPEC_DIR / "name-intcol.yaml")
col_tfm
# ColumnTransformer(
#     transformers=[("std", StandardScaler(), [0, 5]), ("robust", RobustScaler(), [2])]
# )

feat_xs = col_tfm.fit_transform(x_df)
# array([[ 1.51861254,  0.80899739,  0.20143885],
#        ...,


# ================================================
# Example: pipeline
col_tfm = ft.build(ft.SPEC_DIR / "pipeline.yaml")
col_tfm
# ColumnTransformer(
#     transformers=[
#         ("standardscaler", StandardScaler(), ["alcohol", "total_phenols"]),
#         (
#             "pipeline",
#             Pipeline(
#                 steps=[
#                     ("simpleimputer", SimpleImputer(strategy="constant")),
#                     ("robustscaler", RobustScaler()),
#                 ]
#             ),
#             ["ash"],
#         ),
#     ]
# )

feat_xs = col_tfm.fit_transform(x_df)
feat_xs
# array([[ 1.51861254,  0.80899739,  0.20143885],
#        ...,


# ================================================
# Example: ColumnTransformer settings
col_tfm = ft.build(ft.SPEC_DIR / "settings.yaml")
col_tfm
# ColumnTransformer(
#     n_jobs=-1,
#     transformers=[
#         ("standardscaler", StandardScaler(), ["alcohol", "total_phenols"]),
#         ("robustscaler", RobustScaler(), ["ash"]),
#     ],
# )

feat_xs = col_tfm.fit_transform(x_df)
feat_xs
# array([[ 1.51861254,  0.80899739,  0.20143885],
#        ...,


# ================================================
# Example: full X, y feature transform with save/load
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


# ================================================
# Example: use helper to suggest spec
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
