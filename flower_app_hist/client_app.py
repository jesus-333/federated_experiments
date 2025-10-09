"""
@author: Alberto Zancanaro (Jesus)
@organization: Luxembourg Centre for Systems Biomedicine (LCSB)
@contact : alberto.zancanaro@uni.lu
@date: September 2025
"""
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Imports

import numpy as np
import pandas as pd

from collections.abc import Iterable

from flwr.client import ClientApp
from flwr.common import Context, Message, MetricRecord, RecordDict

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# import warnings
# warnings.filterwarnings("ignore", category = UserWarning)

# Flower ClientApp
app = ClientApp()

@app.query()
def query(msg : Message, context : Context):
    """Construct histogram of local dataset and report to `ServerApp`."""
    
    """
    Example of msg and context
    Message(metadata=Metadata(run_id=8348619855088795003, message_id='f12007fe252d28bb0f86e7216a657c78588403b20763751b87d4d27a5835486f', src_node_id=1, dst_node_id=11631752601818482563, reply_to_message_id='', group_id='1', created_at=1759151430.732682, ttl=43200.0, message_type='query', delivered_at=''), content=RecordDict(
      array_records={},
      metric_records={},
      config_records={}
    ))
    Context(run_id=8348619855088795003, node_id=11631752601818482563, node_config={'num-partitions': 2, 'partition-id': 1}, state=RecordDict(
      array_records={},
      metric_records={},
      config_records={}
    ), run_config={'num-server-rounds': 3, 'fraction-sample': 1.0})
    """

    # Get config (node_config)
    # partition_id = context.node_config["partition-id"]
    path_client_data = context.node_config["path_client_data"]

    # Remember that the node config is passed as an argument of flower-supernode command.
    # For an example see https://flower.ai/docs/framework/how-to-run-flower-with-deployment-engine.html#start-two-flower-supernodes

    # Get config (from the message)
    my_config       = msg.content.config_records["my_config"]
    server_round    = my_config["server_round"]
    bins_variable   = my_config["bins_variable"]
    
    # Get the dataset
    data_hist_all, _     = get_data(path_client_data, bins_variable)
    data_hist_UC, _      = get_data(path_client_data, bins_variable, 'UC')
    data_hist_CD, _      = get_data(path_client_data, bins_variable, 'CD')
    data_hist_control, _ = get_data(path_client_data, bins_variable, 'Control')

    query_results = {}

    if server_round == 0 :
        # Min-max computation
        query_results["min"] = np.min(data_hist_all).item()
        query_results["max"] = np.max(data_hist_all).item()
    elif server_round == 1 :
        # Histogram computation

        # Get bins from the message sent by the server
        bins = my_config["bins"]

        # Compute histogram
        freqs_all, _     = np.histogram(data_hist_all, bins = bins)
        freqs_UC, _      = np.histogram(data_hist_UC, bins = bins)
        freqs_CD, _      = np.histogram(data_hist_CD, bins = bins)
        freqs_control, _ = np.histogram(data_hist_control, bins = bins)
    
        # Save the histogram
        query_results["histogram_all"]     = freqs_all.tolist()
        query_results["histogram_UC"]      = freqs_UC.tolist()
        query_results["histogram_CD"]      = freqs_CD.tolist()
        query_results["histogram_control"] = freqs_control.tolist()

        # Save average and std
        query_results["average_all"]     = np.mean(data_hist_all).item()
        query_results["std_all"]         = np.std(data_hist_all).item()
        query_results["average_UC"]      = np.mean(data_hist_UC).item()
        query_results["std_UC"]          = np.std(data_hist_UC).item()
        query_results["average_CD"]      = np.mean(data_hist_CD).item()
        query_results["std_CD"]          = np.std(data_hist_CD).item()
        query_results["average_control"] = np.mean(data_hist_control).item()
        query_results["std_control"]     = np.std(data_hist_control).item()
    else :
        raise ValueError(f"Server round {server_round} not supported.")

    reply_content = RecordDict({"query_results": MetricRecord(query_results)})

    return Message(reply_content, reply_to = msg)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def get_data(path_client_data : str, bins_variable : str, class_to_filter : str = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Load the data for a specific client from a CSV file.

    Parameters
    ----------
    path_client_data : str
        The path to the CSV file containing the client data.
    bins_variable : str
        The name of the column containing the histogram data.
    class_to_filter : str
        The class to keep in the data. If None, all classes are kept. Default is None.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        A tuple containing:
        - data_hist : np.ndarray
            The histogram data for the specified client.
        - labels_per_sample : np.ndarray
            The labels for each sample in the histogram data.
    """
    
    # Load the dataset and get the data
    dataset_client = pd.read_csv(path_client_data)
    data_hist = dataset_client[bins_variable].to_numpy()
    
    # Get the labels
    labels_per_sample = dataset_client['Diagnosis'].to_numpy()

    # (OPTIONAL) Filter the data to keep only the specified classes
    if class_to_filter is not None :
        
        # Create a boolean index to keep only the specified classes
        idx_to_keep = labels_per_sample == class_to_filter

        # Filter the data and labels
        data_hist = data_hist[idx_to_keep]
        labels_per_sample = labels_per_sample[idx_to_keep]

    return data_hist, labels_per_sample
