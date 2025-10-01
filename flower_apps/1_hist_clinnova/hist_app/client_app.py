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
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner

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
    partition_id = context.node_config["partition-id"]
    path_client_data = context.node_config["path_client_data"]

    # Remember that the node config is passed as an argument of flower-supernode command.
    # For an example see https://flower.ai/docs/framework/how-to-run-flower-with-deployment-engine.html#start-two-flower-supernodes

    # Get config (from the message)
    my_config       = msg.content.config_records["my_config"]
    server_round    = my_config["server_round"]
    bins_variable   = my_config["bins_variable"]
    class_to_filter = my_config["class_to_filter"] if "class_to_filter" in my_config else None
    
    # Get the dataset
    data_hist, labels_per_sample = get_data(partition_id, path_client_data, bins_variable, class_to_filter)

    query_results = {}

    if server_round == 0 :
        # Min-max computation
        query_results["min"] = np.min(data_hist).item()
        query_results["max"] = np.max(data_hist).item()
    elif server_round == 1 :
        # Histogram computation

        # Get bins from the message sent by the server
        bins = my_config["bins"]

        # Compute histogram
        freqs, _ = np.histogram(data_hist, bins = bins)
    
        # Save the histogram
        query_results["histogram"] = freqs.tolist()
    
        # Save average and std
        query_results["average"] = np.mean(data_hist).item()
        query_results["std"]     = np.std(data_hist).item()
    else :
        raise ValueError(f"Server round {server_round} not supported.")

    reply_content = RecordDict({"query_results": MetricRecord(query_results)})

    return Message(reply_content, reply_to = msg)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def get_data(partition_id : int, path_client_data : str, bins_variable : str, class_to_filter : Iterable[str] = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Load the data for a specific client from a CSV file.

    Parameters
    ----------
    partition_id : int
        The ID of the client (partition) to load the data for.
    path_client_data : str
        The path to the CSV file containing the client data.
    bins_variable : str
        The name of the column containing the histogram data.
    class_to_filter : Iterable[str], optional
        The classes to keep in the data. If None, all classes are kept. Default is None.

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
        idx_to_keep = np.ones(len(data_hist)) == 0

        # Loop over the classes to keep
        for label in class_to_filter :
            tmp_idx = labels_per_sample == label
            idx_to_keep = np.logical_or(idx_to_keep, tmp_idx)

        # Filter the data and labels
        data_hist = data_hist[idx_to_keep]
        labels_per_sample = labels_per_sample[idx_to_keep]

    return data_hist, labels_per_sample
