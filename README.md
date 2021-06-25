<p align="left">
<img width=15% src="https://dai.lids.mit.edu/wp-content/uploads/2018/06/Logo_DAI_highres.png" alt=“AutoBazaar” />
<i>An open source project from Data to AI Lab at MIT.</i>
</p>

[![Development Status](https://img.shields.io/badge/Development%20Status-2%20--%20Pre--Alpha-yellow)](https://pypi.org/search/?c=Development+Status+%3A%3A+2+-+Pre-Alpha)
[![PyPi](https://img.shields.io/pypi/v/autobazaar.svg)](https://pypi.python.org/pypi/autobazaar)
[![Tests](https://github.com/MLBazaar/AutoBazaar/workflows/Run%20Tests/badge.svg)](https://github.com/MLBazaar/AutoBazaar/actions?query=workflow%3A%22Run+Tests%22+branch%3Amaster)
[![Downloads](https://pepy.tech/badge/autobazaar)](https://pepy.tech/project/autobazaar)

# AutoBazaar

* License: [MIT](https://github.com/MLBazaar/AutoBazaar/blob/master/LICENSE)
* Development Status: [Pre-Alpha](https://pypi.org/search/?c=Development+Status+%3A%3A+2+-+Pre-Alpha)
* Documentation: https://MLBazaar.github.io/AutoBazaar/
* Homepage: https://github.com/MLBazaar/AutoBazaar
* Paper: [here][ml-bazaar-paper]

## Overview

*AutoBazaar* is an AutoML system created using [The Machine Learning Bazaar](https://mlbazaar.github.io),
a research project and framework for building ML and AutoML systems by the [Data To AI Lab](https://dai.lids.mit.edu) at MIT.
See [below](#citing-autobazaar) for more references.

It comes in the form of a Python library which can be used directly inside any other Python
project, as well as a CLI which allows searching for pipelines to solve a problem directly
from the command line.

# Install

## Requirements

AutoBazaar has been developed and tested on [Python 3.6 and 3.7](https://www.python.org/downloads/)

Also, although it is not strictly required, the usage of a
[virtualenv](https://virtualenv.pypa.io/en/latest/) is highly recommended in order to avoid
interfering with other software installed in the system where AutoBazaar is run.

## Install with pip

The easiest and recommended way to install AutoBazaar is using
[pip](https://pip.pypa.io/en/stable/):

```bash
pip install autobazaar
```

This will pull and install the latest stable release from [PyPI](https://pypi.org/).

If you want to install from source or contribute to the project please read the
[Contributing Guide](https://MLBazaar.github.io/AutoBazaar/contributing.html#get-started).

# Data Format

AutoBazaar works with datasets in the [D3M Schema Format](https://github.com/mitll/d3m-schema)
as input.

This dataset schema, developed by MIT Lincoln Labs Laboratory for DARPA's Data-Driven Discovery
of Models (D3M) Program, requires the data to be in plainly readable formats such as CSV files or
JPG images, and to be set within a folder hierarchy alongside some metadata specifications
in JSON format, which include information about all the data contained, as well as the problem
that we are trying to solve.

For more details about the schema and about how to format your data to be compliant with it,
refer to the [Schema Documentation](https://github.com/mitll/d3m-schema/tree/master/documentation)

As an example, you can browse some datasets which have been included in this repository for
demonstration purposes:
- [185_baseball](https://github.com/MLBazaar/AutoBazaar/tree/master/input/185_baseball): Single Table Regression
- [196_autoMpg](https://github.com/MLBazaar/AutoBazaar/tree/master/input/196_autoMpg): Single Table Classification

Additionally, you can find a collection with ~450 datasets already in the D3M Schema in the [ML Bazaar Task Suite](https://mlbazaar.github.io/#datasets-and-tasks) (please request access [here](https://mlbazaar.github.io/#how-can-i-request-access-to-the-datasets)).

# Quickstart

In this short tutorial we will guide you through a series of steps that will help you getting
started with AutoBazaar using its CLI command `abz`.

For more details about its usage and the available options, please execute `abz --help`
on your command line.

## 1. Prepare your Data

Make sure to have your data prepared in the [Data Format](#data-format) explained above inside
and uncompressed folder in a filesystem directly accessible by AutoBazaar.

In order to check, whether your dataset is available and ready to use, you can execute
the `abz list` subcommand.
If your dataset is in a different place than inside a folder called `data` within your
current working directory, add the `-i` argument to your command indicating
the path to the folder that contains your dataset.

Assuming that the data is inside a folder called `input` within your current folder,
you can run:

```bash
$ abz list -i path/to/your/datasets/folder
```

The output should be a table which includes the details of all the datasets found inside
the indicated directory:

```
             data_modality                task_type task_subtype             metric size_human  train_samples
dataset
185_baseball  single_table           classification  multi_class            f1Macro       148K           1073
196_autoMpg   single_table               regression   univariate   meanSquaredError        32K            298
30_personae           text           classification       binary                 f1       1,4M            116
32_wikiqa      multi_table           classification       binary                 f1       4,9M          23406
60_jester     single_table  collaborative_filtering               meanAbsoluteError        44M         880719
```

> :bulb: If you see an error saying that `No matching datasets found`, please review your
> dataset format and make sure you have indicated the right path.

For the rest of this quickstart, we will be using the `185_baseball` dataset that you can
find inside the [input folder](https://github.com/MLBazaar/AutoBazaar/tree/master/input)
contained in this repository.

## 2. Start the search process

Once your data is ready, you can start the AutoBazaar search process using the `abz search`
command. To do this, you will need to provide again the path to where your datasets are contained, as
well as the name of the datasets that you want to process.

Without further configuration, the search process will evaluate only the default pipeline without performing additional tuning iterations on it.

```bash
abz search -i path/to/your/datasets/folder name_of_your_dataset
```

In order to start a real search process, you will need to provide at least one of the
following additional options:

* `-b, --budget`:
    Maximum number of tuning iterations to perform.
* `-c, --checkpoints`:
    Comma separated string containing the different checkpoints, in seconds,
    where the best pipeline so far must be stored and evaluated against the
    test dataset. There must be no spaces between the checkpoint times. For
    example, to store the best pipeline every 10 minutes until 30 minutes have
    elapsed, you would use the option `-c 600,1200,1800`. If checkpoints are
    provided, the system will terminate at the time of the final checkpoint.
* `-t, --timeout`:
    Maximum time for the system to run, in seconds. Ignored if checkpoints are
    given.

For example, to search over the `185_baseball` dataset for a 30 second period, evaluating the
best pipeline so far every 10 seconds, but with a maximum of 10 tuning iterations, we would
use the following command:

```bash
abz search 185_baseball -c10,20,30 -b10
```

For further details about the available options, run `abz search --help`.

## 3. Explore the results

Once AutoBazaar has finished searching for the best pipeline, a table will be printed
to stdout with a summary of the best pipeline found for each dataset.
If multiple checkpoints were provided, details about the best pipeline in each checkpoint
will also be included.

The output will be a table similar to this one:

```
                                          pipeline     score      rank  cv_score   metric data_modality       task_type task_subtype    elapsed  iterations  load_time  trivial_time  fit_time    cv_time error  step
dataset
185_baseball  fce28425-e45c-4620-9d3c-d329b8684bea  0.316961  0.682957  0.317043  f1Macro  single_table  classification  multi_class  10.024457         0.0   0.011041      0.026212       NaN        NaN  None  None
185_baseball  f7428924-79ee-439d-bc32-998a9efea619  0.675132  0.390927  0.609073  f1Macro  single_table  classification  multi_class  21.412262         1.0   0.011041      0.026212   9.99484        NaN  None  None
185_baseball  397780a5-6bf6-48c9-9a85-06b0d08c5a9d  0.675132  0.357361  0.642639  f1Macro  single_table  classification  multi_class  31.712946         2.0   0.011041      0.026212   9.99484  12.618179  None  None
```

Alternatively, a `-r` option can be passed with the name of a CSV file, and the results will
be stored there:

```bash
abz search 185_baseball -c10,20,30 -b10 -r results.csv
```

## What's next?

For more details about AutoBazaar and all its possibilities and features, please check the
[project documentation site](https://MLBazaar.github.io/AutoBazaar/)!

## Citing AutoBazaar

If you use AutoBazaar for your research, please consider citing
[our paper about ML Bazaar][ml-bazaar-paper]:

```bibtex
@inproceedings{smith2020machine,
    author = "Smith, Micah J. and Sala, Carles and Kanter, James Max and Veeramachaneni, Kalyan",
    title = "The {{Machine Learning Bazaar}}: {{Harnessing}} the {{ML Ecosystem}} for {{Effective System Development}}",
    booktitle = "Proceedings of the 2020 {{ACM SIGMOD International Conference}} on {{Management}} of {{Data}}",
    year = "2020",
    pages = "785--800",
    publisher = "{Association for Computing Machinery}",
    address = "{Portland, OR, USA}",
    doi = "10.1145/3318464.3386146",
    isbn = "978-1-4503-6735-6",
    language = "en",
    series = "{{SIGMOD}} '20"
}
```

[ml-bazaar-paper]: https://doi.org/10.1145/3318464.3386146
