import os
import sys
from setuptools import find_packages, setup
from setuptools.command.test import test as TestCommand

test_args = [
    '--verbose',
    '--capture=sys',
    '--log-level=INFO',
    '--log-cli-level=INFO',
    '--log-file-level=INFO',
    '--cov-report=html',
    '--cov-report=term',
    '--cov=feature_transform',
    'test',
]


class PyTest(TestCommand):
    user_options = [('pytest-args=', 'a', 'Arguments to pass to py.test')]

    def initialize_options(self):
        os.environ['ENV'] = 'test'
        TestCommand.initialize_options(self)
        self.pytest_args = test_args

    def run_tests(self):
        import pytest
        errno = pytest.main(self.pytest_args)
        sys.exit(errno)


setup(
    name='feature_transform',
    version='0.4.1',
    description='Build feature transformer by specifying transformation.',
    long_description='https://github.com/kengz/feature_transform',
    keywords='feature_transform',
    url='https://github.com/kengz/feature_transform',
    author='kengz',
    author_email='kengzwl@gmail.com',
    packages=find_packages(),
    install_requires=[
        'dill>=0.3.4',
        'loguru>=0.5.0',
        'numpy>=1.17.3',
        'pandas>=1.0.5',
        'pydash>=4.7.6',
        'pyyaml>=5.3.0',
        'scikit-learn>=0.24.0',
    ],
    zip_safe=False,
    include_package_data=True,
    dependency_links=[],
    extras_require={},
    classifiers=[],
    tests_require=[
        'autopep8==1.5.3',
        'flake8==3.8.3',
        'pytest==5.4.1',
        'pytest-cov==2.8.1',
    ],
    test_suite='test',
    cmdclass={'test': PyTest},
)
