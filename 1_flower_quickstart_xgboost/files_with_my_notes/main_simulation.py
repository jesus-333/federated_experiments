"""
Implementation of the tutorial: https://flower.ai/docs/framework/tutorial-quickstart-xgboost.html
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Imports

import toml
import xgboost as xgb

import flwr
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner

import server

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Settings

partition_id = 3
random_seed = 42

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

strategy = server.FedXgbBagging()

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Function definition

def train_test_split(partition, test_fraction, seed):
    """Split the data into train and validation set given split rate."""
    train_test = partition.train_test_split(test_size=test_fraction, seed=seed)
    partition_train = train_test["train"]
    partition_test = train_test["test"]

    num_train = len(partition_train)
    num_test = len(partition_test)

    return partition_train, partition_test, num_train, num_test


def transform_dataset_to_dmatrix(data):
    """Transform dataset to DMatrix format for xgboost."""
    x = data["inputs"]
    y = data["label"]
    new_data = xgb.DMatrix(x, label = y)
    return new_data

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Load (HIGGS) dataset and partition.
# We use a small subset (num_partitions=20) of the dataset for demonstration to speed up the data loading process.
partitioner = IidPartitioner(num_partitions = 20)
fds = FederatedDataset(dataset="jxie/higgs", partitioners = {"train": partitioner})

# Load the partition for this `partition_id`
partition = fds.load_partition(partition_id, split = "train")
partition.set_format("numpy")

# Train/test splitting
train_data, valid_data, num_train, num_val = train_test_split(
    partition, test_fraction = 0.2, seed = random_seed
)

# Reformat data to DMatrix for xgboost
# More info about DMatrix here : https://xgboost.readthedocs.io/en/stable/python/python_api.html#xgboost.DMatrix
train_dmatrix = transform_dataset_to_dmatrix(train_data)
valid_dmatrix = transform_dataset_to_dmatrix(valid_data)
