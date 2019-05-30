<p align="left">
<img width=15% src="https://dai.lids.mit.edu/wp-content/uploads/2018/06/Logo_DAI_highres.png" alt=“AutoBazaar” />
<i>An open source project from Data to AI Lab at MIT.</i>
</p>

# AutoBazaar

- License: MIT
- Documentation: https://HDI-Project.github.io/AutoBazaar/
- Homepage: https://github.com/HDI-Project/AutoBazaar

# Overview

AutoBazaar is an AutoML system created to execute the experiments associated with the
[The Machine Learning Bazaar Paper: Harnessing the ML Ecosystem for Effective System
Development](https://arxiv.org/pdf/1905.08942.pdf)
by the [Human-Data Interaction (HDI) Project](https://hdi-dai.lids.mit.edu/) at LIDS, MIT.

# Install

## Requirements

**AutoBazaar** has been developed and tested on [Python 3.5 and 3.6](https://www.python.org/downloads/)

Also, although it is not strictly required, the usage of a
[virtualenv](https://virtualenv.pypa.io/en/latest/) is highly recommended in order to avoid
interfering with other software installed in the system where **AutoBazaar** is run.

These are the minimum commands needed to create a virtualenv using python3.6 for **AutoBazaar**:

```bash
pip install virtualenv
virtualenv -p $(which python3.6) autobazaar-venv
```

Afterwards, you have to execute this command to have the virtualenv activated:

```bash
source autobazaar-venv/bin/activate
```

Remember about executing it every time you start a new console to work on **AutoBazaar**!

## Install with pip

After creating the virtualenv and activating it, we recommend using
[pip](https://pip.pypa.io/en/stable/) in order to install **AutoBazaar**:

```bash
pip install autobazaar
```

This will pull and install the latest stable release from [PyPi](https://pypi.org/).

## Install from source

Alternatively, with your virtualenv activated, you can clone the repository and install it from
source by running `make install` on the `stable` branch:

```bash
git clone git@github.com:HDI-Project/AutoBazaar.git
cd AutoBazaar
git checkout stable
make install
```

For development, you can use `make install-develop` instead in order to install all
the required dependencies for testing and code linting.

# Data Format

TODO: Briefly explain D3M Data Format here and add a link to its schema.

## Datasets Collection

You can find a collection of datasets in the D3M format in the
[d3m-data-dai S3 Bucket in AWS](https://d3m-data-dai.s3.amazonaws.com/index.html).

TODO: Also add a link to the D3M datasets repository.

# Quickstart

In this short tutorial we will guide you through a series of steps that will help you getting
started with **AutoBazaar**.

TODO

# Credits

AutoBazaar is an Open Source project from the Data to AI Lab at MIT built by the following team:

* Carles Sala <csala@csail.mit.edu>
* Micah Smith <micahs@mit.edu>
* Max Kanter <max.kanter@gmail.com>
* Kalyan Veeramachaneni <kalyan@mit.edu>
