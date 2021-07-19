# Command line access to PAPYRUS mongoDB.
# See https://ultimagen.atlassian.net/wiki/spaces/AG/pages/1428914739/Papyrus+Metrics+Infrastructure+Proof+of+Concept#Command-Line-Access
# for documentation

import pymongo
import pandas as pd
import warnings
from typing import Optional
import os
import json


def initialize_client() -> pymongo.MongoClient:
    '''Initializes pymongo client with the access string that is read from PAPYRUS_ACCESS_STRING
    environmental variable.

    Parameters:
    -----------
    None

    Returns
    -------
    pymongo.MongoClient
    '''

    myclient = pymongo.MongoClient(os.environ['PAPYRUS_ACCESS_STRING'])
    return myclient


DISABLE_PAPYRUS_ACCESS = False
if 'PAPYRUS_ACCESS_STRING' in os.environ:
    myclient = initialize_client()
    mydb = myclient["pipelines"]
    mycollection = mydb["pipelines"]
else:
    warnings.warn(
        "Define PAPYRUS_ACCESS_STRING environmental variable to enable access to Papyrus")
    warnings.warn(
        "Example: export PAPYRUS_ACCESS_STRING=mongodb+srv://[user]:[passwd]@testcluster.jm2x3.mongodb.net/test")
    DISABLE_PAPYRUS_ACCESS = True


def query_database(query: dict) -> list:
    '''Querying pipelines database. For easy access
    Define PAPYRUS_ACCESS_STRING environmental variable
    Example: export PAPYRUS_ACCESS_STRING=mongodb+srv://[user]:[passwd]@testcluster.jm2x3.mongodb.net/test

    Parameters
    ----------
    query : dict
        Pymongo query (dictionary)

    Returns
    -------
    list
        List of documents
    '''

    assert not DISABLE_PAPYRUS_ACCESS, "Database access not available through PAPYRUS_ACCESS_STRING"
    return [x for x in mycollection.find(query)]


DEFAULT_METRICS_TO_REPORT = ['AlignmentSummaryMetrics', 'Contamination', 'DuplicationMetrics',
                             'GcBiasDetailMetrics', 'GcBiasSummaryMetrics', 'QualityYieldMetrics',
                             'RawWgsMetrics', 'WgsMetrics', 'stats_coverage', 'short_report_/all_data']


def metrics2df(doc: dict, metrics_to_report: list = DEFAULT_METRICS_TO_REPORT) -> pd.DataFrame:
    f'''Converts metrics document to pandas dataframe

    Parameters
    ----------
    doc: dict
        Single document from mongodb (dictionary, output of query_database)
    metrics_to_report: list
        which metrics should be reported (default {", ".join(DEFAULT_METRICS_TO_REPORT)})

    Returns
    -------
    pd.DataFrame    
        1xn dataframe with two level index on columns: (metric_type, metric_name)
    '''

    md = pd.DataFrame(pd.DataFrame(doc['metadata']).query(
        'workflowEntity=="sample"').loc['entityType']).T
    md.index = [0]
    md = pd.concat({'metadata': md}, axis=1)
    result = [(x, pd.read_json(json.dumps(doc['metrics'][x]), orient='table'))
              for x in doc['metrics'] if x in metrics_to_report]
    result = [x for x in result if x[1].shape[0] == 1]
    result_df = pd.concat((md, pd.concat(dict(result), axis=1)), axis=1).set_index(
        ("metadata", "workflowId"))
    result_df.index = result_df.index.rename("workflowId")
    return result_df


def inputs2df(doc: dict) -> pd.DataFrame:
    '''Returns a dataframe of inputs and outputs metadata

    Parameters
    ----------
    doc: dict
        Single document from mongoDB

    Returns
    -------
    pd.DataFrame
        Dataframe of inputs and outputs combined (single row)
    '''
    md = pd.DataFrame(doc['metadata']).query(
        'workflowEntity=="sample"').loc['entityType']
    inputs = pd.Series(doc['inputs'])
    outputs = pd.Series(doc['outputs'])
    return pd.DataFrame(pd.concat((md, inputs, outputs))).T.set_index("workflowId")
