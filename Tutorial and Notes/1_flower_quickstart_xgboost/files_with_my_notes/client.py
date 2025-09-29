"""
Implementation of the tutorial: https://flower.ai/docs/framework/tutorial-quickstart-xgboost.html
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Imports

from logging import INFO
import xgboost as xgb

import flwr
from flwr.common.typing import FitRes, FitIns, EvaluateIns, EvaluateRes
from flwr.common.typing import Status, Parameters, Code
from flwr.common.config import unflatten_dict
from flwr.common.context import Context
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Settings

partition_id = 3
random_seed = 42

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Function definition

def train_test_split(partition, test_fraction, random_seed):
    """Split the data into train and validation set given split rate."""
    train_test = partition.train_test_split(test_size=test_fraction, seed = random_seed)
    partition_train = train_test["train"]
    partition_test = train_test["test"]

    num_train = len(partition_train)
    num_test = len(partition_test)

    return partition_train, partition_test, num_train, num_test


def transform_dataset_to_dmatrix(data):
    """
    Transform dataset to DMatrix format for xgboost.
    """
    
    # Get data and label
    x = data["inputs"]
    y = data["label"]

    # Convert in Dmatrix
    new_data = xgb.DMatrix(x, label = y)

    return new_data

def load_data(partition_id, num_clients, random_seed = 42):
    """
    Load partition HIGGS data.
    """

    # Only initialize `FederatedDataset` once
    global fds
    if fds is None:
        partitioner = IidPartitioner(num_partitions = num_clients)
        fds = FederatedDataset(
            dataset = "jxie/higgs",
            partitioners = {"train": partitioner},
        )

    # Load the partition for this `partition_id`
    partition = fds.load_partition(partition_id, split = "train")
    partition.set_format("numpy")

    # Train/test splitting
    train_data, valid_data, num_train, num_val = train_test_split(
        partition, test_fraction = 0.2, seed = random_seed
    )

    # Reformat data to DMatrix for xgboost
    flwr.common.log(INFO, "Reformatting data...")
    train_dmatrix = transform_dataset_to_dmatrix(train_data)
    valid_dmatrix = transform_dataset_to_dmatrix(valid_data)

    return train_dmatrix, valid_dmatrix, num_train, num_val


def replace_keys(input_dict, match = "-", target = "_"):
    """
    Recursively replace match string with target string in dictionary keys.
    """

    new_dict = {}
    for key, value in input_dict.items():
        new_key = key.replace(match, target)
        if isinstance(value, dict):
            new_dict[new_key] = replace_keys(value, match, target)
        else:
            new_dict[new_key] = value
    return new_dict

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Client. Define Flower Client and client_fn

# Client class
class FlowerClient(flwr.client.Client):
    def __init__(self,
        train_dmatrix, valid_dmatrix,
        num_train, num_val, num_local_round,
        params ):
        
        # Save data
        self.train_dmatrix = train_dmatrix
        self.valid_dmatrix = valid_dmatrix

        self.num_train = num_train
        self.num_val = num_val
        self.num_local_round = num_local_round

        # Client model parameter
        self.params = params

    def fit(self, ins: FitIns) -> FitRes:
        # Get the actual round of federated training
        global_round = int(ins.config["global_round"])
        if global_round == 1:
            # First round local training 

            # Create the tree from scratch
            # More info here https://xgboost.readthedocs.io/en/stable/python/python_api.html#xgboost.train
            bst = xgb.train(
                self.params,
                self.train_dmatrix,
                num_boost_round = self.num_local_round,
                evals=[(self.valid_dmatrix, "validate"), (self.train_dmatrix, "train")],
            )
        else:
            # All other rounds

            # Create the object for the xgboost model. More info here https://xgboost.readthedocs.io/en/stable/python/python_api.html#xgboost.Booster
            # From the link: Booster is the model of xgboost, that contains low level routines for training, prediction and evaluation.
            bst = xgb.Booster(params = self.params)

            # Get the global model
            global_model = bytearray(ins.parameters.tensors[0])

            # Load global model into booster
            bst.load_model(global_model)

            # Local training
            bst = self._local_boost(bst)

        # Save model
        local_model = bst.save_raw("json")
        local_model_bytes = bytes(local_model)

        return FitRes(
            status = Status(
                code = Code.OK,
                message = "OK",
            ),
            parameters = Parameters(tensor_type = "", tensors = [local_model_bytes]),
            num_examples = self.num_train,
            metrics = {},
        )

    def _local_boost(self, bst_input):
        """
        Train for the local xgboost model after round 1
        """

        # Update trees based on local training data.
        for i in range(self.num_local_round):
            bst_input.update(self.train_dmatrix, bst_input.num_boosted_rounds())

        # Bagging: extract the last N=num_local_round trees for sever aggregation
        bst = bst_input[
            bst_input.num_boosted_rounds() - self.num_local_round : bst_input.num_boosted_rounds()
        ]

        return bst

    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        # Load global model
        bst = xgb.Booster(params = self.params)
        para_b = bytearray(ins.parameters.tensors[0])
        bst.load_model(para_b)

        # Run evaluation
        eval_results = bst.eval_set(
            evals=[(self.valid_dmatrix, "valid")],
            iteration = bst.num_boosted_rounds() - 1,
        )
        auc = round(float(eval_results.split("\t")[1].split(":")[1]), 4)

        return EvaluateRes(
            status = Status(
                code = Code.OK,
                message = "OK",
            ),
            loss = 0.0,
            num_examples = self.num_val,
            metrics = {"AUC": auc},
        )

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

def client_fn(context: Context):
    """
    Function that create and return a client.
    """

    # Load model and data
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    train_dmatrix, valid_dmatrix, num_train, num_val = load_data( partition_id, num_partitions)

    cfg = replace_keys(unflatten_dict(context.run_config))
    num_local_round = cfg["local_epochs"]

    # Return Client instance
    return FlowerClient(
        train_dmatrix,
        valid_dmatrix,
        num_train,
        num_val,
        num_local_round,
        cfg["params"],
    )
