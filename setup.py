#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import find_packages, setup

with open('README.md') as readme_file:
    readme = readme_file.read()

with open('HISTORY.md') as history_file:
    history = history_file.read()

install_requires = [
    'absl-py==0.4.0',
    'astor==0.7.1',
    'baytune==0.2.1',
    'boto==2.48.0',
    'boto3==1.9.27',
    'botocore==1.12.28',
    'certifi==2018.8.13',
    'chardet==3.0.4',
    'click==6.7',
    'cloudpickle==0.4.0',
    'cycler==0.10.0',
    'dask==0.18.2',
    'decorator==4.3.0',
    'distributed==1.22.1',
    'docutils==0.14',
    'featuretools==0.3.1',
    'future==0.16.0',
    'gast==0.2.0',
    'grpcio==1.12.1',
    'h5py==2.8.0',
    'HeapDict==1.0.0',
    'idna==2.6',
    'iso639==0.1.4',
    'jmespath==0.9.3',
    'Keras==2.1.6',
    'Keras-Applications==1.0.6',
    'Keras-Preprocessing==1.0.5',
    'kiwisolver==1.0.1',
    'langdetect==1.0.7',
    'lightfm==1.15',
    'matplotlib==2.2.3',
    'mit-d3m==0.1.1',
    'mlblocks==0.2.3',
    'mlprimitives==0.1.3',
    'msgpack==0.5.6',
    'networkx==2.1',
    'nltk==3.3',
    'numpy==1.15.2',
    'opencv-python==3.4.2.17',
    'pandas==0.23.4',
    'Pillow==5.1.0',
    'protobuf==3.6.1',
    'psutil==5.4.7',
    'pymongo==3.7.2',
    'pyparsing==2.2.0',
    'python-dateutil==2.7.3',
    'python-louvain==0.10',
    'pytz==2018.5',
    'PyWavelets==0.5.2',
    'PyYAML==3.12',
    'requests==2.20.0',
    's3fs==0.1.5',
    's3transfer==0.1.13',
    'scikit-image==0.14.0',
    'scikit-learn==0.20.0',
    'scipy==1.1.0',
    'six==1.11.0',
    'sortedcontainers==2.0.4',
    'setuptools==39.1.0',
    'tblib==1.3.2',
    'tensorboard==1.11.0',
    'tensorflow==1.11.0',
    'termcolor==1.1.0',
    'toolz==0.9.0',
    'tornado==5.1',
    'tqdm==4.24.0',
    'urllib3==1.23',
    'Werkzeug==0.14.1',
    'xgboost==0.72.1',
    'zict==0.1.3',
]

tests_require = [
    'pytest>=3.4.2',
    'pytest-cov>=2.6.0',
]

setup_requires = [
    'pytest-runner>=2.11.1',
]

development_requires = [
    # general
    'bumpversion>=0.5.3',
    'pip>=10.0.0',
    'watchdog>=0.8.3',

    # docs
    'm2r>=0.2.0',
    'Sphinx>=1.7.1',
    'sphinx_rtd_theme>=0.2.4',
    'autodocsumm>=0.1.10',

    # style check
    'flake8>=3.5.0',
    'isort>=4.3.4',

    # fix style issues
    'autoflake>=1.1',
    'autopep8>=1.3.5',

    # distribute on PyPI
    'twine>=1.10.0',
    'wheel>=0.30.0',

    # Advanced testing
    'tox>=2.9.1',
    'coverage>=4.5.1',
]

setup(
    author='MIT Data To AI Lab',
    author_email='dailabmit@gmail.com',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    description='The Machine Learning Bazaar',
    entry_points={
        'console_scripts': [
            'abz=autobazaar.__main__:main'
        ]
    },
    extras_require={
        'dev': development_requires + tests_require,
        'tests': tests_require,
    },
    include_package_data=True,
    install_requires=install_requires,
    license='MIT license',
    long_description=readme + '\n\n' + history,
    long_description_content_type='text/markdown',
    keywords='automl machine learning hyperparameters tuning classification regression autobazaar',
    name='autobazaar',
    packages=find_packages(include=['autobazaar', 'autobazaar.*']),
    python_requires='>=3.4',
    setup_requires=setup_requires,
    test_suite='tests',
    tests_require=tests_require,
    url='https://github.com/HDI-project/AutoBazaar',
    version='0.1.1-dev',
    zip_safe=False,
)
