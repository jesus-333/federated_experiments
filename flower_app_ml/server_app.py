"""
A Flower `ServerApp` that train a machine learning algorithm.
Currently implemented algorithm are LDA, SVM, neural network, k-means.

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

from flwr.common import ArrayRecord, ConfigRecord, Context, Message, MessageType, RecordDict
from flwr.common.logger import log
from flwr.server import Grid, ServerApp
from flwr.serverapp import strategy

import support

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Flower ServerApp

app = ServerApp()

@app.main()
def main(grid: Grid, context: Context) -> None:
    """
    This `ServerApp` construct a histogram from partial-histograms reported by the `ClientApp`s.
    """

    path_server_config = context.run_config['path_server_config']
    server_config = toml.load(path_server_config)

    fields_to_use_for_train = support.read_txt_list(server_config['path_file_with_fields_to_use_for_the_train'])

    # Load server data
    x_server, y_server, _ = support.get_data(server_config['path_server_data'], fields_to_use_for_train)
    
    # Path to save the final results
    path_to_save = server_config['path_to_save'] if 'path_to_save' in server_config else './results/'

    # Get config for the ml algorithm
    ml_model_config = server_config['ml_algorithm_config']
    
    # Dictionary used to communicate with the clients
    my_config = dict(
        ml_model_name = server_config['ml_model_name'],
        ml_model_config = ml_model_config,
        fields_to_use_for_train = fields_to_use_for_train
    )

    # Create ml model
    ml_model = support.get_ml_model(my_config['ml_model_name'], ml_model_config)
    log(INFO, f"ML Model created: {ml_model}")

    # Setting initial parameters (it is required by flower) and convert them in an ArrayRecord representation
    support.set_initial_params(my_config['ml_model_name'], ml_model, 3, x_server.shape[1])
    arrays = ArrayRecord(support.get_model_params(ml_model))
    
    # Create FL strategy
    fl_strategy = strategy.FedAvg()
    
    # Federated training
    result = fl_strategy.start(
        grid = grid,
        initial_arrays = arrays,
        num_rounds = server_config['num_rounds'],
        train_config = ConfigRecord(my_config),
    )
    
    # Get the results
    # Note that the function to_numpy_ndarrays() return the ArrayRecord as a list of NumPy ndarray.
    params_final = result.arrays.to_numpy_ndarrays()
    support.set_model_params(my_config['ml_model_name'], ml_model, params_final)

    # Save the final weights of the model
    with open(f'{path_to_save}/final_params_{my_config["ml_model_name"]}.pkl', "wb") as f : pickle.dump(params_final, f)

    # Get the model weights of the single node
    n_nodes = server_config['n_nodes']
    node_ids = get_node_ids(grid, n_nodes)
    list_params_per_node = get_model_weights_from_clients(grid, node_ids, my_config)
    
    # Save the model weights of the single node
    for i in range(n_nodes) :
        with open(f'{path_to_save}/trained_params_{my_config["ml_model_name"]}_node_{node_ids[i]}.pkl', "wb") as f :
            pickle.dump(list_params_per_node[i], f)

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

def send_and_receive_data(grid: Grid, node_ids: list[int], server_round: int, my_config : dict = None) -> list[Message] | None:
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

def get_model_weights_from_clients(grid: Grid, node_ids : list[int], my_config : dict = None, max_number_of_attempts : int = 10) -> list :
    """
    Implemented to obtain the model weights of a single nodes outside the FL process. From what I see the FedAvg strategy only return the finale model weights after the last round.
    For the PoC I also need the model weights of each node to show the difference between the models trained on different datasets.

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
    list_params_per_node : list
        List containing the model weights of each node.
        The order of the nodes is the same as in the node_ids list.
    """

    n_attempts = 0
    while (True) :
        results = send_and_receive_data(grid, node_ids, server_round = 0, my_config = my_config)

        if results is not None :
            break
        else :
            n_attempts += 1
            log(INFO, f"Error in receiving data from clients. Attempt {n_attempts}/{max_number_of_attempts}")
            if n_attempts >= max_number_of_attempts :
                raise Exception(f"Error in receiving data from clients during round {my_config['server_round']}. Maximum number of attempts ({max_number_of_attempts}) reached")
            time.sleep(2)

    list_params_per_node = []

    for rep in results :
        # Get the content of the message
        # Note that the key "query_results" is not a predefined key from the Flower framework. It is just a key used in the client app.
        # If you want you could use whatever key you want, as long as it is the same in the client and server app.
        query_results = rep.content["query_results"]
        
        # Get the model weights
        params = query_results['model_weights']
        list_params_per_node.append(params)

    return list_params_per_node

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

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
