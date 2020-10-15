import json
import uuid

import numpy as np
import pandas as pd
from d3mdm.local import LocalManager
from d3mdm.splitter import add_dataset_splits, write_csv


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
def _generate_dataframe_dict(
    name,
    df,
    target,
    taskType,
    taskMetric,
    outputpath,
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

    # add splits
    add_dataset_splits(data, name)

    PATH = outputpath
    lm = LocalManager(PATH)
    lm.write(data, base_dir=name)


def csv2d3m(
    name: str,
    csvpath: str,
    target: str,
    taskType: str,
    taskMetric: str,
    outputpath: str,
    d3mindex: str = None,
    text_column: str = None,
):
    df = pd.read_csv(csvpath)
    _generate_dataframe_dict(
        name, df, target, taskType, taskMetric, outputpath, d3mindex=d3mindex,
        text_column=text_column)


if __name__ == '__main__':
    from argparse import ArgumentParser
    from sys import exit

    from mit_d3m.metrics import METRICS_DICT

    parser = ArgumentParser(
        prog='csv2d3m',
        description='utility for converting CSV to D3M format',
    )
    parser.add_argument(
        '--dataset-name',
        required=True, type=str,
        help='some name to give the dataset')
    parser.add_argument(
        '--csv-path',
        required=True, type=str,
        help='path to CSV file')
    parser.add_argument(
        '--target-colname',
        required=True, type=str,
        help='name of column in CSV file identifying prediction target')
    parser.add_argument(
        '--task-type',
        required=True, type=str,
        help='task type (see D3M schema)')
    parser.add_argument(
        '--task-metric',
        required=True, choices=METRICS_DICT.keys(),
        help='task metric')
    parser.add_argument(
        '--output-path',
        default='./input',
        help='path to output dir (must exist)')
    parser.add_argument(
        '--index-colname',
        default=None,
        help='name of index in CSV file identifying unique '
             'observations (defaults to DataFrame index')
    args = parser.parse_args()
    exit(
        csv2d3m(
            args.dataset_name, args.csv_path, args.target_colname,
            args.task_type, args.task_metric, args.output_path,
            d3mindex=args.index_colname))
