from pathlib import Path

import sklearn

from feature_transform.helper import suggest  # noqa
from feature_transform.validator.spec import Spec, build  # noqa

SPEC_DIR = Path(__file__).parent / "example" / "spec"


def register_class(cls: type):
    """Register a class in sklearn"""
    # first check for conflict
    if hasattr(sklearn, cls.__name__):
        raise ValueError(f"Module {cls.__name__} already exists in sklearn")
    setattr(sklearn, cls.__name__, cls)
