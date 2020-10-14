# -*- coding: utf-8 -*-

import json
import logging
import os
import pathlib
import tempfile
import uuid
from collections import defaultdict
from datetime import datetime
from io import StringIO

import funcy as fy
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

LOGGER = logging.getLogger(__name__)


def encode_score(scorer, expected, observed):
    if expected.dtype == 'object':
        le = LabelEncoder()
        expected = le.fit_transform(expected)
        observed = le.transform(observed)

    return scorer(expected, observed)


def ensure_dir(directory):
    """Create diretory if it does not exist yet."""
    if not os.path.exists(directory):
        os.makedirs(directory)


def make_dumpable(params, datetimes=False):
    """Get nested dicts of params to allow json dumping.

    Also work around this: https://github.com/HDI-Project/BTB/issues/79
    And convert numpy types to primitive types.

    Optionally dump datetimes to ISO format.

    Args:
        params (dict):
            Params dictionary with tuples as keys.
        datetimes (bool):
            whether to convert datetimes to ISO strings or not.

    Returns:
        dict:
            Dumpable params as a tree of dicts and nested sub-dicts.
    """
    nested_params = defaultdict(dict)
    for (block, param), value in params.items():
        if isinstance(value, np.integer):
            value = int(value)

        elif isinstance(value, np.floating):
            value = float(value)

        elif isinstance(value, np.ndarray):
            value = value.tolist()

        elif isinstance(value, np.bool_):
            value = bool(value)

        elif value == 'None':
            value = None

        elif datetimes and isinstance(value, datetime):
            value = value.isoformat()

        nested_params[block][param] = value

    return nested_params


def _walk(document, transform):
    if not isinstance(document, dict):
        return document

    new_doc = dict()
    for key, value in document.items():
        if isinstance(value, dict):
            value = _walk(value, transform)
        elif isinstance(value, list):
            value = [_walk(v, transform) for v in value]

        new_key, new_value = transform(key, value)
        new_doc[new_key] = new_value

    return new_doc


def remove_dots(document):
    """Replace dots with dashes in all the keys from the dictionary."""
    return _walk(document, lambda key, value: (key.replace('.', '-'), value))


def restore_dots(document):
    """Replace dashes with dots in all the keys from the dictionary."""
    return _walk(document, lambda key, value: (key.replace('-', '.'), value))


def make_keras_picklable():
    """Make the keras models picklable."""

    import keras.models  # noqa: lazy import slow dependencies

    def __getstate__(self):
        model_str = ""
        with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=True) as fd:
            keras.models.save_model(self, fd.name, overwrite=True)
            model_str = fd.read()

        return {'model_str': model_str}

    def __setstate__(self, state):
        with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=True) as fd:
            fd.write(state['model_str'])
            fd.flush()
            try:
                model = keras.models.load_model(fd.name)

            except ValueError:
                from keras.applications import mobilenet
                from keras.utils.generic_utils import CustomObjectScope
                scope = {
                    'relu6': mobilenet.relu6,
                    'DepthwiseConv2D': mobilenet.DepthwiseConv2D
                }
                with CustomObjectScope(scope):
                    model = keras.models.load_model(fd.name)

        self.__dict__ = model.__dict__

    cls = keras.models.Model
    cls.__getstate__ = __getstate__
    cls.__setstate__ = __setstate__


def write_csv(df: pd.DataFrame, dataset: dict, csv_path: str):
    # dataset appears to be a dict representing a file system structure
    folder = dataset
    parts = pathlib.Path(csv_path).parts
    folder = fy.get_in(folder, parts[:-1])

    buf = StringIO()
    df.to_csv(buf)
    folder[parts[-1]] = buf.getvalue().encode('utf-8')
    buf.close()


class LocalManager(object):
    def __init__(self, datasets_path, skip_sublevels=False):
        self.datasets_path = datasets_path
        self.skip_sublevels = skip_sublevels

    @classmethod
    def load_folder(cls, folder, prefixes):
        data = dict()
        for name in os.listdir(folder):
            path = os.path.join(folder, name)
            if any(prefix in path or path in prefix for prefix in prefixes):
                if os.path.isdir(path):
                    data[name] = cls.load_folder(path, prefixes)

                else:
                    with open(path, "rb") as f:
                        data[name] = f.read()

        return data

    def load(self, dataset_name, raw=False):
        dataset_path = os.path.join(self.datasets_path, dataset_name)
        LOGGER.info("Loading dataset %s", dataset_path)
        if raw:
            problem = dataset_name + "_problem"
            problems = [name for name in os.listdir(dataset_path) if problem in name]
            dataset = dataset_name + "_dataset"

            if self.skip_sublevels:
                # restrict the dataset sublevels to datasetDoc.json and tables
                dataset_sublevels = [
                    os.path.join(dataset, "tables"),
                    os.path.join(dataset, "datasetDoc.json"),
                ]
                prefixes = problems + dataset_sublevels

            else:
                prefixes = problems + [dataset]

        else:
            prefixes = os.listdir(dataset_path)

        prefixes = [os.path.join(dataset_path, prefix) for prefix in prefixes]

        return self.load_folder(dataset_path, prefixes)

    def write(self, dataset, base_dir="", root=True):

        full_base_dir = os.path.join(self.datasets_path, base_dir)
        if root:
            LOGGER.info("Writing dataset %s", full_base_dir)

        if not os.path.exists(full_base_dir):
            os.makedirs(full_base_dir)

        for key, value in dataset.items():
            path = os.path.join(base_dir, key)
            if isinstance(value, dict):
                self.write(value, path, False)

            else:
                path = os.path.join(self.datasets_path, path)
                LOGGER.debug("Writing file %s", path)
                with open(path, "wb") as f:
                    f.write(value)

    def datasets(self):
        return list(sorted(os.listdir(self.datasets_path)))

    def exists(self, dataset_name):
        dataset_path = os.path.join(self.datasets_path, dataset_name)
        return os.path.exists(dataset_path)


def _extract_text(df, name, text_column):
    texts = dict()

    for text in df[text_column]:
        filename = str(uuid.uuid4()) + ".txt"
        texts[filename] = str(text).encode()

    del df[text_column]
    df["raw_text_file"] = texts.keys()
    return texts, df


def _analyze_column(col, df):
    # char_len = col.astype(str).str.len()
    # aux = df[char_len == char_len.max()][col.name]

    is_float = col.dtype == np.dtype("float64")
    is_int = col.dtype == np.dtype("int64")
    # is_obj = col.dtype == np.dtype("object")
    is_bool = col.dtype == np.dtype("bool")

    if is_float:
        if all(col.values.astype(int)):
            return "integer"

        else:
            return "float"

    if is_bool:
        return "boolean"

    if is_int:
        return "integer"

    return "string"


def _generate_columns(col_name, col_type, col_index, target=-1, tab_index="d3mIndex"):

    data = {
        "colIndex": col_index,
        "colName": col_name,
        "colType": col_type,
        "role": ["attribute"],
    }

    if col_name == tab_index:
        data["role"] = ["index"]

    if col_type == "text":
        data["refersTo"] = {"resID": "0", "resObject": "item"}

        data["colType"] = "string"

    if col_index == target:
        data["role"] = ["suggestedTarget"]

    return data


def _get_datadoc(file_name: str, df: pd.DataFrame, target: str):

    columns = list()

    for col in df:
        if df[col].dtype in ["float64", "int64"]:
            df[col].fillna(0, inplace=True)

        else:
            df[col].fillna("", inplace=True)

        col_type = _analyze_column(df[col], df)

        col_index = df.columns.get_loc(col)
        target_index = df.columns.get_loc(target)
        columns.append(
            _generate_columns(col, col_type, col_index, target_index))

    data = {
        "about": {
            "datasetID": file_name + "_dataset",
            "datasetName": "",
            "description": "",
            "citation": "",
            "license": "Creative Commons",
            "source": "",
            "sourceURI": "",
            "approximateSize": "",
            "datasetSchemaVersion": "3.0",
            "redacted": False,
            "datasetVersion": "1.0",
        },
        "dataResources": [
            {
                "resID": "0",
                "resPath": "tables/learningData.csv",
                "resType": "table",
                "resFormat": ["text/csv"],
                "isCollection": False,
                "columns": columns,
            }
        ],
    }

    return json.dumps(data, indent=4).encode()


def _get_problemdoc(file_name, df, target, taskType, taskMetric):

    data = {
        "about": {
            "problemID": file_name + "_problem",
            "problemName": "",
            "problemDescription": "",
            "taskType": taskType,
            "taskSubType": "",
            "problemSchemaVersion": "3.0",
            "problemVersion": "1.0",
        },
        "inputs": {
            "data": [
                {
                    "datasetID": file_name + "_dataset",
                    "targets": [
                        {
                            "targetIndex": 0,
                            "resID": "0",
                            "colIndex": df.columns.to_list().index(target),
                            "colName": target,
                        }
                    ],
                }
            ],
            "dataSplits": {
                "method": "holdOut",
                "testSize": 0.2,
                "numRepeats": 0,
                "splitsFile": "dataSplits.csv",
            },
            "performanceMetrics": [{"metric": taskMetric}],
        },
        "expectedOutputs": {"predictionsFile": "predictions.csv"},
    }

    return json.dumps(data, indent=4).encode()


def _generate_structure(
    name: str, df: pd.DataFrame, target: str, taskType: str, taskMetric: str
) -> dict:
    return {
        name + "_dataset": {
            "tables": {"learningData.csv": ""},
            "datasetDoc.json": _get_datadoc(name, df, target),
        },
        name + "_problem": {
            "problemDoc.json": _get_problemdoc(name, df, target, taskType, taskMetric),
        }
    }


# Datasets_path
def generate_dataframe_dict(
    name,
    df,
    target,
    taskType,
    taskMetric,
    d3mindex=None,
    text_column=None,
    other_df=None,
):
    df = df.copy()

    if d3mindex:
        df.rename(columns={d3mindex: "d3mIndex"}, inplace=True)

    else:
        df_columns = list(df.columns)
        df["d3mIndex"] = df.index
        df = df[["d3mIndex"] + df_columns]

    if text_column:
        # raw_text_file
        text_dict, df = _extract_text(df, name, text_column)
        data = _generate_structure(name, df, target, taskType, taskMetric)
        data[name + "_dataset"]["text"] = text_dict

    else:
        data = _generate_structure(name, df, target, taskType, taskMetric)

    if other_df:
        data_doc = json.loads(data[name + "_dataset"]["datasetDoc.json"])
        x = 0
        for df_name, dataframe, tabindex in other_df:
            columns = list()
            x = x + 1
            for col in dataframe:
                if dataframe[col].dtype in ["float64", "int64"]:
                    dataframe[col].fillna(0, inplace=True)
                else:
                    dataframe[col].fillna("", inplace=True)
                col_type = _analyze_column(dataframe[col], dataframe)
                col_index = dataframe.columns.get_loc(col)
                columns.append(
                    _generate_columns(col, col_type, col_index, tab_index=tabindex)
                )

            data_doc["dataResources"].append(
                {
                    "resID": str(x),
                    "resPath": "tables/" + df_name,
                    "resType": "table",
                    "resFormat": ["text/csv"],
                    "isCollection": False,
                    "columns": columns,
                }
            )
            csvFile = name + "_dataset/tables/" + df_name
            write_csv(dataframe, data, csvFile)

        data[name + "_dataset"]["datasetDoc.json"] = json.dumps(
            data_doc, indent=4
        ).encode()

    # write to the file tree the dataframe
    learnData = name + "_dataset/tables/learningData.csv"
    write_csv(df, data, learnData)

    PATH = "_datasets"

    lm = LocalManager(PATH)
    lm.write(data, base_dir=name)
