#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import find_packages, setup

with open('README.md') as readme_file:
    readme = readme_file.read()

with open('HISTORY.md') as history_file:
    history = history_file.read()

install_requires = [
    'boto3>=1.14,<1.14.45',
    'botocore<1.17.45,>=1.17.44',
    'mlblocks>=0.3.2,<0.4',
    'mlprimitives>=0.2.2,<0.3',
    'baytune>=0.2.1,<0.3',
    'mit-d3m>=0.2.1,<0.3',
    'Keras>=2.1.6,<2.4',
    'scikit-learn<0.21,>=0.20.0',
    'pandas<0.25,>=0.23.4',
    'numpy<1.17,>=1.15.2',
    'python-dateutil<2.8.1,>=2.1',
]

tests_require = [
    'pytest>=3.4.2',
    'pytest-cov>=2.6.0',
    'rundoc>=0.4.3,<0.5',
]

setup_requires = [
    'pytest-runner>=2.11.1',
]

development_requires = [
    # general
    'pip>=9.0.1',
    'bumpversion>=0.5.3,<0.6',
    'watchdog>=0.8.3,<0.11',

    # docs
    'm2r>=0.2.0,<0.3',
    'autodocsumm>=0.1.10,<0.2',
    'Sphinx>=1.7.1,<3',
    'sphinx_rtd_theme>=0.2.4,<0.5',

    # style check
    'flake8>=3.7.7,<4',
    'isort>=4.3.4,<5',

    # fix style issues
    'autoflake>=1.1,<2',
    'autopep8>=1.4.3,<2',

    # distribute on PyPI
    'twine>=1.10.0,<4',
    'wheel>=0.30.0',

    # Advanced testing
    'coverage>=4.5.1,<6',
    'tox>=2.9.1,<4',
    'rundoc>=0.4.3,<0.5',
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
        'test': tests_require,
    },
    include_package_data=True,
    install_requires=install_requires,
    license='MIT license',
    long_description=readme + '\n\n' + history,
    long_description_content_type='text/markdown',
    keywords='automl machine learning hyperparameters tuning classification regression autobazaar',
    name='autobazaar',
    packages=find_packages(include=['autobazaar', 'autobazaar.*']),
    python_requires='>=3.6,<3.8',
    setup_requires=setup_requires,
    test_suite='tests',
    tests_require=tests_require,
    url='https://github.com/HDI-project/AutoBazaar',
    version='0.2.2.dev0',
    zip_safe=False,
)
