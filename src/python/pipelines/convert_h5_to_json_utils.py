import json
import os
import pandas as pd
import tables


def get_h5_keys(h5_filename: str):
    with pd.HDFStore(h5_filename) as store:
        keys = store.keys()
        return keys


def should_skip_h5_key(key: str, ignored_h5_key_substring: str):
    if ignored_h5_key_substring is not None:
        return ignored_h5_key_substring in key


def preprocess_h5_key(key: str):
    result = key
    if result[0] == "/":
        result = result[1:]
    return result


def preprocess_columns(dataframe):
    """Handle multiIndex/ hierarchical .h5 - concatenate the columns for using it as single string in JSON."""
    if hasattr(dataframe, 'columns'):
        MULTI_INDEX_SEPARATOR = "___"
        if isinstance(dataframe.columns, pd.core.indexes.multi.MultiIndex):
            dataframe.columns = [MULTI_INDEX_SEPARATOR.join(col).rstrip(
                MULTI_INDEX_SEPARATOR) for col in dataframe.columns.values]


def preprocess_json_for_mongodb(json_string: str):
    """Replace characters which are invalid in MongoDB."""
    result = json_string
    result = result.replace("%", "_Percent")
    result = result.replace("$", "__")
    return result


def log(str: str):
    print(str)


def convert_h5_to_json(input_h5_filename: str, root_element: str, mongodb_preprocessing: bool, ignored_h5_key_substring: str):
    """Convert an .h5 metrics file to .json with control over the root element and the processing

    Parameters
    ----------
    input_h5_filename: str
        Input h5 file name

    root_element: str
        Root element of the returned json

    mongodb_preprocessing: bool
        Do preprocessing to adapt the JSON file to MongoDB

    ignored_h5_key_substring: str
        A way to filter some of the keys using substring match

    Returns
    -------
    str
        The result json string includes the schema (the types) of the metrics as well as the metrics themselves.

    """

    new_json_dict = {root_element: {}}
    h5_keys = get_h5_keys(input_h5_filename)
    for h5_key in h5_keys:
        if should_skip_h5_key(h5_key, ignored_h5_key_substring):
            log(f'Skipping: {h5_key}')
            continue
        log(f'Processing: {h5_key}')
        df = pd.read_hdf(input_h5_filename, h5_key)
        preprocess_columns(df)
        df_to_json = df.to_json(orient="table")
        json_dict = json.loads(df_to_json)
        new_json_dict[root_element][preprocess_h5_key(h5_key)] = json_dict

    json_string = json.dumps(new_json_dict, indent=4)
    if mongodb_preprocessing:
        json_string = preprocess_json_for_mongodb(json_string)
    return json_string 
