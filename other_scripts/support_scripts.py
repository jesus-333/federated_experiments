"""
@author: Alberto Zancanaro (Jesus)
@organization: Luxembourg Centre for Systems Biomedicine (LCSB)
@contact : alberto.zancanaro@uni.lu
@date: September 2025
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

import numpy as np

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def get_idx_to_split_data_V1(n_elements : int, percentage_split : float, seed : int = None):
    """
    Given a number of elements (n_elements) create an array with number from 0 to n_elements - 1 and split it (randomly) in two lists.
    The size of the two list is determined by the percentage_split parameter. The first list will be have size x = int(percentage_split * n_elements) while the second will have size y = n_elements - x
    The procedure can be "deterministic" if the seed parameter is passed to the function.
    """
    
    # Check input parameter
    if n_elements <= 1 : raise ValueError("n_elements must be greater than 1. Current value is {}".format(n_elements))
    if percentage_split <= 0 or percentage_split >= 1 : raise ValueError("percentage_split must be between 0 and 1. Current value is {}".format(percentage_split))

    # Use of the seed for reproducibility
    if seed is not None : np.random.seed(seed)

    # Create idx vector
    idx = np.random.permutation(n_elements)
    size_1 = int(n_elements * percentage_split)
    
    return idx[0:size_1], idx[size_1:]

def random_split_array(input_aray, percentage_split : float, seed : None) :
    """
    Given an input array, split it in two arrays with a specific percentage.
    The seed parameter can be used to make the split deterministic.
    If percentage_split is 0 or 1 the same array will be returned together with an empty list.
    If percentage_split is 0 the return value will be empty list, original array
    If percentage_split is 1 the return value will be original array, empty list
    """

    # Check input parameter
    if percentage_split < 0 or percentage_split > 1 : raise ValueError("percentage_split must be between 0 and 1 (included). Current value is {}".format(percentage_split))

    if percentage_split == 0 :
        return [], input_aray[:]
    elif percentage_split == 1 :
        return input_aray[:], []
    else :
        # Use of the seed for reproducibility 
        if seed is not None : np.random.seed(seed)

        # Split the array
        idx_1, idx_2 = get_idx_to_split_data_V1(len(input_aray), percentage_split)

        return input_aray[idx_1], input_aray[idx_2]

def get_idx_to_split_data_V2(n_elements : int, percentage_split_list : list, seed : int = None):
    """
    Given a number of elements (n_elements) create an array with number from 0 to n_elements - 1 and split it (randomly) in n lists.
    Each of the n list will have a percentage of elements determined by the percentage_split_list parameter. The sum of the elements in percentage_split_list must be equal to 1.

    E.g. n_elements = 100, percentage_split_list = [0.6, 0.3, 0.1]. The first list will have 60 elements, the second 30 and the third 10.
    The procedure can be "deterministic" if the seed parameter is passed to the function.

    Parameters
    ----------
    n_elements : int
        Number of elements to split
    percentage_split_list : list
        List with the percentage of elements for each split
    seed : int
        Seed for reproducibility. Default is None.
    """
    
    # Check input parameter
    # Note that check for 0.9999999999999999 and 0.9999999999999998, 1.0000000000000001, 1.0000000000000002 is due to the float precision
    possible_sum = [1, 0.9999999999999999, 0.9999999999999998, 1.0000000000000001, 1.0000000000000002]
    if n_elements <= 1 : raise ValueError("n_elements must be greater than 1. Current value is {}".format(n_elements))
    if np.sum(percentage_split_list) not in possible_sum : raise ValueError("The sum of the elements in percentage_split_list must be equal to 1. Current sum is {}".format(np.sum(percentage_split_list)))

    # Use of the seed for reproducibility
    if seed is not None : np.random.seed(seed)

    # Create idx vector
    idx_to_split = np.arange(n_elements).astype(int)

    # Create splits with the idx
    idx_list = []
    for i in range(len(percentage_split_list) - 1) :
        percentage = percentage_split_list[i]
        
        if i == 0 :
            actual_percentage = percentage
        else :
            size_split = int(percentage * n_elements)
            actual_percentage = size_split / len(idx_to_split)

        idx_to_save, idx_to_split = random_split_array(idx_to_split, actual_percentage, seed)
        idx_list.append(idx_to_save)

    # The last split is the remaining idx
    idx_list.append(idx_to_split)

    return idx_list

def get_idx_to_split_data_V3(labels_list : list, percentage_split_list : list, seed : int = None):
    """
    Given a list of labels (labels_list) create an array with number from 0 to len(labels_list) - 1 and split it in n lists.
    Each of the n list will have a percentage of elements determined by the percentage_split_list parameter. The sum of the elements in percentage_split_list must be equal to 1.
    The proportion of the labels in the splits is preserved, i.e. if the label 'A' is 10% of the dataset, in each split there will be 10% elements with label 'A'.
    Of course this is possibile only if there are enough elements for each label in the dataset.
    
    Parameters
    ----------
    labels_list : list
        List with the labels
    percentage_split_list : list
        List with the percentage of elements for each split
    seed : int
        Seed for reproducibility. Default is None.
    """

    # Check input parameter
    # Note that check for 0.9999999999999999 is due to the float precision
    possible_sum = [1, 0.9999999999999999, 0.9999999999999998, 0.9999999999999997, 0.9999999999999996]
    if np.sum(percentage_split_list) not in possible_sum : raise ValueError("The sum of the elements in percentage_split_list must be equal to 1. Current sum is {}".format(np.sum(percentage_split_list)))

    # Use of the seed for reproducibility
    if seed is not None : np.random.seed(seed)

    # Get the unique labels
    unique_labels = np.unique(labels_list)

    # Convert the labels_list to numpy array
    labels_list = np.asarray(labels_list)
    
    # Create list with list of indices
    idx_list = []
    for i in range(len(percentage_split_list)) : idx_list.append([])

    for unique_label in unique_labels :
        # Get the idx for the unique label
        idx_for_current_label = np.where(labels_list == unique_label)[0]

        # Get split for current indices
        tmp_split_for_current_label = get_idx_to_split_data_V2(len(idx_for_current_label), percentage_split_list, seed)
    
        # For each unique values of the labels add the indices to the various list
        for i in range(len(percentage_split_list)) :
            idx_list[i] = idx_list[i] + list(idx_for_current_label[tmp_split_for_current_label[i]])
    
    return idx_list

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Federated data split

def split_data_for_clients(data, percentage_split_per_client : list, seed : int = None, labels = None, keep_labels_proportion : int = False) :
    """
    Split the data (and labels if provided) among the clients according to the percentage_split_per_client.
    The number of clients is supposed to be equal to the length of percentage_split_per_client.
    If keep_labels_proportion is True, the original proportion of labels is kept for each client. E.g. if the original data has 10% of label 1 and 90% of label 0, each client will have 10% of label 1 and 90% of label 0.
    Note that keep_labels_proportion works only if there are enough samples for each label for each client.
    """
    
    # Check input parameters
    # Note that check for the number different than 1 are necessary due to floating point precision "error"
    possible_sum = [1, 0.9999999999999999, 0.9999999999999998, 1.0000000000000001, 1.0000000000000002]
    if np.sum(percentage_split_per_client) not in possible_sum : raise ValueError(f"The sum of the elements in percentage_split_list must be equal to 1. Current sum is {np.sum(percentage_split_per_client)}")
    if keep_labels_proportion and labels is None : raise ValueError("keep_labels_proportion is True but labels is None")
    
    # Get indices for each client
    if keep_labels_proportion :
        idx_list = get_idx_to_split_data_V2(len(data), percentage_split_per_client, seed)
    else :
        idx_list = get_idx_to_split_data_V3(labels, percentage_split_per_client, seed)
    
    # Variables to store data (and labels) for each client
    data_per_client = []
    labels_per_client = [] if labels is not None else None

    # Split data (and labels) for each client
    for i in range(len(percentage_split_per_client)) :
        idx_client = idx_list[i]
        data_per_client.append(data[idx_client])
        if labels is not None : labels_per_client.append(labels[idx_client])
    
    # Return values
    if labels is not None :
        return data_per_client, labels_per_client
    else :
        return data_per_client

def split_data_for_clients_uniformly(data, n_client : int, seed : int = None, labels = None, keep_labels_proportion : int = False) :
    """
    Split the data (and labels if provided) uniformly among the clients.
    If keep_labels_proportion is True, the original proportion of labels is kept for each client. E.g. if the original data has 10% of label 1 and 90% of label 0, each client will have 10% of label 1 and 90% of label 0.
    Note that keep_labels_proportion works only if there are enough samples for each label for each client. Also if class are highly unbalanced, and/or the number of sample for specific class is not divisible by n_client, the split is not perfectly uniform.
    """

    data_per_client = []
    labels_per_client = [] if labels is not None else None

    percentage_split_per_client = [1 / n_client] * n_client

    if labels is not None :
        data_per_client, labels_per_client = split_data_for_clients(data, percentage_split_per_client, seed, labels, keep_labels_proportion)
        return data_per_client, labels_per_client
    else :
        data_per_client = split_data_for_clients(data, percentage_split_per_client, seed)
        return data_per_client

def check_split_correctness(original_data, original_label, data_per_client, labels_per_client) :
    """
    Check that the division of data and labels per client is correct.
    The input parameters original_data and original_label must be the same used for data and labels for the function split_data_for_clients (or split_data_for_clients_uniformly).
    The input paramenters data_per_client and labels_per_client must be the output of the function.

    Note that this function simply check if each data in data_per_client has the correct label in labels_per_client.
    Yeah, I know that this is not very useful, especially if split_data_for_clients (or split_data_for_clients_uniformly) works correctly but I add for my peace of mind.
    """
    
    # For each sample in the original data create a dictionary where the key is the sample and the value is the label
    data_to_label = dict()
    for i in range(len(original_data)) : data_to_label[original_data[i]] = original_label[i]
    
    for i in range(len(data_per_client)) :
        data_specific_client = data_per_client[i]
        labels_specific_client = labels_per_client[i]
        for j in range(len(data_specific_client)) :
            current_data = data_specific_client[j]
            current_label = labels_specific_client[j]

            if data_to_label[current_data] != current_label :
                raise ValueError(f"Data and label do not match for client {i} and sample {j}. Actual data is {current_data}. Actual label is {current_label} and expected label is {data_to_label[current_data]}")

    print("Everything seems correct")
