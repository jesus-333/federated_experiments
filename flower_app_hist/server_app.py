"""
A Flower `ServerApp` that constructs a histogram from clients data.

@author: Alberto Zancanaro (Jesus)
@organization: Luxembourg Centre for Systems Biomedicine (LCSB)
@contact : alberto.zancanaro@uni.lu
@date: September 2025
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Imports

import numpy as np
import os
import pickle
import time
import toml

from collections.abc import Iterable
from logging import INFO

from flwr.common import Context, Message, MessageType, RecordDict, ConfigRecord
from flwr.common.logger import log
from flwr.server import Grid, ServerApp

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Flower ServerApp

app = ServerApp()

@app.main()
def main(grid: Grid, context: Context) -> None:
    """
    This `ServerApp` construct a histogram from partial-histograms reported by the `ClientApp`s.
    """

    path_server_config = context.run_config['path_server_config'] if 'path_server_config' in context.run_config else './server_config.toml'
    server_config = toml.load(path_server_config)

    # General settings
    min_nodes              = server_config['min_nodes'] if 'min_nodes' in server_config else server_config['n_nodes']
    max_number_of_attempts = server_config['max_number_of_attempts'] if 'max_number_of_attempts' in server_config else 10
    
    # Variable used to create the hist bins
    max, min = None, None
    n_bins = server_config['n_bins'] if 'n_bins' in server_config else 10
    
    # Predefined min and max could be used. By default they are None
    # If both are provided the round 0 for min-max computation will be skipped, otherwise the missing value will be computed
    predefined_min = server_config['predefined_min'] if 'predefined_min' in server_config else None
    predefined_max = server_config['predefined_max'] if 'predefined_max' in server_config else None
    
    # Path to save the final histogram
    path_to_save = server_config['path_to_save'] if 'path_to_save' in server_config else './results/'
    
    # Dictionary used to communicate with the clients
    my_config = dict(
        server_round = -1,
        bins_variable = server_config['bins_variable'],
        bins_distribution = server_config['bins_distribution'] if 'bins_distribution' in server_config else 'uniform'
    )

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # Min and max computation round (round 0)

    if predefined_min is None or predefined_max is None :

        log(INFO, "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
        log(INFO, "START ROUND for min and max computation (round 0)")

        # Update config for round 0
        my_config['server_round'] = 0
        
        # Get all node ids
        # Note that this id will be used both for round 0 and round 1
        node_ids_round = get_node_ids(grid, min_nodes)
        
        # Get the min and max from the clients
        results_round_zero = get_data_from_clients(grid, node_ids_round, my_config, max_number_of_attempts)

        # Compute global min and max
        min, max = compute_min_max_federation(results_round_zero)
        
        # Overwrite min or max if predefined values are provided
        if predefined_min is not None : min = predefined_min
        if predefined_max is not None : max = predefined_max

    else :
        min, max = predefined_min, predefined_max

    log(INFO, f"Computed global min: {min}" if predefined_min is None else f"Using predefined min: {min}")
    log(INFO, f"Computed global max: {max}" if predefined_max is None else f"Using predefined max: {max}")
    log(INFO, "END ROUND for min and max computation (round 0)")

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # Histogram computation rounds (round 1)

    log(INFO, "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    log(INFO, "START ROUND for histogram computation (round 1)")
    
    # Compute bins
    if my_config['bins_distribution'] == 'uniform' :
        bins = np.linspace(min, max, n_bins + 1)
    elif my_config['bins_distribution'] == 'logarithmic' :
        if min == 0 : 
            # Avoid issues with log scale if min is 0
            min = 1e-10
            bins = np.geomspace(min, max, n_bins + 1)
        elif min < 0 and max > 0 :
            # Avoid issues with log scale if min is negative and max is positive
            # TODO Eventually implement a solution for this case
            # Not necessary fot the PoC since the min value in all dataset, in the worst case, is 0
            pass
        else :
            bins = np.geomspace(min, max, n_bins + 1)

    log(INFO, f"Using bins: {bins}")

    # Update config for round 1
    my_config['server_round'] = 1
    my_config['bins'] = list(bins)
    
    # Get the partial histograms from the clients
    results_round_one = get_data_from_clients(grid, node_ids_round, my_config, max_number_of_attempts)
    
    # Compute final histogram
    final_hist_per_label, samples_mean_per_label, samples_std_per_label = compute_hist(n_bins, results_round_one)
    log(INFO, f"Final histogram (all samples): {final_hist_per_label['all']}")

    # Save results
    for label in ['all', 'UC', 'CD', 'control'] :
        save_results(label, my_config, final_hist_per_label[label], samples_mean_per_label[label], samples_std_per_label[label], path_to_save)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Generic functions

def get_node_ids(grid: Grid, min_nodes: int) -> list[int]:
    """
    Loop and wait until enough nodes are available.
    
    Parameters
    ----------
    grid : Grid
        The Flower Grid instance.
        See https://flower.ai/docs/framework/ref-api/flwr.serverapp.Grid.html for more details.
    min_nodes : int
        Minimum number of nodes required.

    Returns
    -------
    list[int]
        List of all node ids.
    """
    
    # List for storing all node ids
    all_node_ids : list[int] = []

    # Loop until enough nodes are available
    while len(all_node_ids) < min_nodes:
        # Fetch all node ids
        all_node_ids = list(grid.get_node_ids())

        # If enough nodes are available, break the loop
        if len(all_node_ids) >= min_nodes:
            break

        # If not enough nodes are available, wait and try again
        log(INFO, "Waiting for nodes to connect...")
        time.sleep(2)

    return all_node_ids

def send_and_receive_data(grid: Grid, node_ids: list[int], server_round: int, my_config : dict = None) :
    """
    Send messages to the specified node ids and wait for all results.

    Parameters
    ----------
    grid : Grid
        The Flower Grid instance.
        See https://flower.ai/docs/framework/ref-api/flwr.serverapp.Grid.html for more details.
    node_ids : list[int]
        List of node ids to which send the messages.
    server_round : int
        The current server round.
    my_config : dict, optional
        Dictionary containing personal configuration to be sent to the clients, by default None.
        If None, an empty dictionary will be sent.

    Returns
    -------
    replies : list[Message] | None
        The results obtained from the clients. They are instances of the Message class.
        See https://flower.ai/docs/framework/ref-api/flwr.common.Message.html for more details about the Message class.
        If an error occurred, None is returned.
    """
    
    # Create messages
    messages = []
    
    # Add other information to message
    recorddict = RecordDict()

    # Add personal configuration to message
    if my_config is not None : recorddict['my_config'] = ConfigRecord(my_config)
    # recorddict['my_config'] = ConfigRecord(my_config if my_config is not None else {})

    for node_id in node_ids:  # one message for each node
        message = Message(
            content = recorddict,
            message_type = MessageType.QUERY,
            dst_node_id = node_id,
            group_id = str(server_round),
        )

        messages.append(message)

        # Some notes about the Message class
        # The message_type can be one of the following : EVALUATE, QUERY, SYSTEM, TRAIN. Based on the type used, a different method will be called in the client.
        # In this case we use QUERY, so the `query` method in ClientApp will be called (With the decorator implementation, it is the function decorated with @app.query).
        # The group_id is used to group messages. In some settings, this is used as the federated learning round.
        # From flower documentation : "The ID of the group to which this message is associated. In some settings, this is used as the federated learning round"

    # Send messages and wait for all results
    replies = grid.send_and_receive(messages)
    log(INFO, "Received %s/%s results", len(replies), len(messages))
    
    # Check for errors
    for rep in replies :
        if rep.has_error():
            return None

    return replies

def get_data_from_clients(grid: Grid, node_ids : list[int], my_config : dict = None, max_number_of_attempts : int = 10) -> list[Message]:
    """
    Use the function `send_and_receive_data` to send messages to the clients and receive their results.
    If an error occurs, the function will retry until the maximum number of attempts is reached.

    Parameters
    grid : Grid
        The Flower Grid instance.
        See https://flower.ai/docs/framework/ref-api/flwr.serverapp.Grid.html for more details.
    node_ids : list[int]
        List of node ids to which send the messages.
    my_config : dict, optional
        Dictionary containing personal configuration to be sent to the clients, by default None.
        If None, an empty dictionary will be sent.
    max_number_of_attempts : int, optional
        Maximum number of attempts to send the messages and receive the results, by default 10.

    Returns
    -------
    results : list[Message]
        The results obtained from the clients. They are instances of the Message class.
        See https://flower.ai/docs/framework/ref-api/flwr.common.Message.html for more details about the Message class.
    """

    n_attempts = 0
    while (True) :
        results = send_and_receive_data(grid, node_ids, server_round = 0, my_config = my_config)

        if results is not None :
            # If no error, break the loop
            break
        else :
            n_attempts += 1
            log(INFO, f"Error in receiving data from clients. Attempt {n_attempts}/{max_number_of_attempts}")
            if n_attempts >= max_number_of_attempts :
                raise Exception(f"Error in receiving data from clients during round {my_config['server_round']}. Maximum number of attempts ({max_number_of_attempts}) reached")
            time.sleep(2)

    return results

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Min-max computation round functions (round 0)

def compute_min_max_federation(results_round_zero: Iterable[Message]) -> tuple[float, float]:
    """
    Compute the global min and max from a list of local mins and maxs.

    Parameters
    ----------
    results_round_zero : Iterable[Message]
        List of messages obtained form the clients.
        See https://flower.ai/docs/framework/ref-api/flwr.common.Message.html for more details about the Message class.

    Returns
    -------
    min : float
        The global min.
    max : float
        The global max.
    """
    
    # Lists for storing all local mins and maxs
    min_list = []
    max_list = []

    for rep in results_round_zero :
        # Get the content of the message
        # Note that the key "query_results" is not a predefined key from the Flower framework. It is just a key used in the client app.
        # If you want you could use whatever key you want, as long as it is the same in the client and server app.
        query_results = rep.content["query_results"]

        # The query_results is an istance of the MetricRecord class.
        # See https://flower.ai/docs/framework/ref-api/flwr.common.MetricRecord.html for more details about the MetricRecord class.
        
        # Append local min and max to the lists
        min_list.append(query_results["min"])
        max_list.append(query_results["max"])

    return min(min_list), max(max_list)

def compute_hist(n_bins : int, results_round_one: Iterable[Message]) -> tuple[dict[str, np.ndarray], dict[str, float], dict[str, float]]:
    """
    Compute the final histogram from a list of client histograms.
    It also compute the mean and std of the data, since the clients also send these values.

    Parameters
    ----------
    n_bins : int
        Number of bins in the histogram.
    results_round_one : Iterable[Message]
        List of messages obtained form the clients. They must contain the local histograms.
        See https://flower.ai/docs/framework/ref-api/flwr.common.Message.html for more details about the Message class.

    Returns
    -------
    final_hist_per_label : dict[str, np.ndarray]
        Dictionary containing the final histogram for each label.
        The keys are 'all', 'UC', 'CD', 'control'.
    samples_mean_per_label : dict[str, float]
        Dictionary containing the mean of the data for each label.
        The keys are 'all', 'UC', 'CD', 'control'.
    samples_std_per_label : dict[str, float]
        Dictionary containing the std of the data for each label.
        The keys are 'all', 'UC', 'CD', 'control'.
    """
    
    # Labels
    labels_list = ['all', 'UC', 'CD', 'control']
    
    # Variable to save the results
    final_hist_per_label = dict()
    samples_mean_per_label = dict()
    samples_std_per_label = dict()

    for label in labels_list :
        # Initialize final histogram
        final_hist = np.zeros(n_bins, dtype = int)
        
        # Lists for storing all local means and stds
        mean_list = []
        std_list  = []

        # Used to compute the weighted average of the mean and std
        n_samples_list = []

        for rep in results_round_one :
            # Get query results
            query_results = rep.content["query_results"]

            # Get local histogram
            local_hist = query_results[f"histogram_{label}"]

            # Sum histograms
            final_hist += np.array(local_hist)

            # Append local mean and std to the lists
            mean_list.append(query_results[f"average_{label}"])
            std_list.append(query_results[f"std_{label}"])

            # Append number of samples to the list
            n_samples_list.append(np.sum(local_hist))

        # Compute mean and std of the data
        # TODO Eventually implement the computation of the std as the pooled std (https://en.wikipedia.org/wiki/Pooled_variance)
        samples_mean = np.average(mean_list, weights = n_samples_list)
        samples_std  = np.average(std_list , weights = n_samples_list)

        # Save results
        final_hist_per_label[label]   = final_hist
        samples_mean_per_label[label] = samples_mean
        samples_std_per_label[label]  = samples_std

    return final_hist_per_label, samples_mean_per_label, samples_std_per_label

def save_results(label : str, info_to_save : dict, final_hist : np.ndarray, samples_mean : float, samples_std : float, path_to_save : str) -> None :
    """
    
    """

    if ":" in info_to_save['bins_variable'] :
        bins_variable_name = info_to_save['bins_variable'].split(":")[1].strip()
    else :
        bins_variable_name = info_to_save['bins_variable']

    path_to_save = os.path.join(path_to_save, bins_variable_name + '/')

    # Add histogram and other info to the dictionary
    info_to_save['histogram'] = final_hist
    info_to_save['mean']      = samples_mean
    info_to_save['std']       = samples_std
    
    # Create folder if it does not exist
    os.makedirs(path_to_save, exist_ok = True)

    # Save info file as a pickle
    with open(path_to_save + f'results_{label}.pkl', 'wb') as f:
        pickle.dump(info_to_save, f)

    # Save info file as a toml
    with open(path_to_save + f'results_{label}.toml', 'w') as f:
        toml.dump(info_to_save, f)

    # Save histogram and bins as numpy arrays
    np.save(path_to_save + f'bins_{label}.npy', np.array(info_to_save['bins']))
    np.save(path_to_save + f'hist_{label}.npy', final_hist)
