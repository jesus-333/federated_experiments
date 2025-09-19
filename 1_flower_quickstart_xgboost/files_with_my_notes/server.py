# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Imports

import xgboost as xgb
import json
from logging import WARNING
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, cast

import flwr
from flwr.common import EvaluateRes, FitRes, Parameters, Scalar, Context
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.client_proxy import ClientProxy

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Define class to implement federated XGBoost with Bagging aggregation.
# Note that this is same source code presented in flower library.
# I simply copy it from the flower quickstart xgboost tutorial (https://flower.ai/docs/framework/tutorial-quickstart-xgboost.html) and add some comments here and there

class FedXgbBagging(flwr.server.strategy.FedAvg):
    """Configurable FedXgbBagging strategy implementation."""

    def __init__(
        self,
        evaluate_function: Optional[
            Callable[
                [int, Parameters, Dict[str, Scalar]],
                Optional[Tuple[float, Dict[str, Scalar]]],
            ]
        ] = None,
        **kwargs: Any,
    ):
        # Note on Option data type : https://stackoverflow.com/questions/51710037/how-should-i-use-the-optional-type-hint

        # Save the evaluation funciton and the global model
        self.evaluate_function = evaluate_function
        self.global_model: Optional[bytes] = None

        # Call parent constructor
        super().__init__(**kwargs)

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using bagging."""
        
        # Do not aggregate if you do not receive results
        if not results:
            return None, {}

        # Do not aggregate if there are failures and failures are not accepted
        # Note that self.accept_failures is inherited from FedAvg. From the documentation : Whether or not accept rounds containing failures. Defaults to True.
        if not self.accept_failures and failures:
            return None, {}

        # Aggregate all the client trees
        global_model = self.global_model
        for _, fit_res in results:
            update = fit_res.parameters.tensors
            for bst in update:
                # The aggregate function is defined below
                global_model = aggregate(global_model, bst)
    
        # Update the server global model
        self.global_model = global_model
        
        # Note that this method overwrites the aggregate_fit of the original FedAvg. 
        # The aggregate method_fit method in general return 2 variables. The first contains the new parameters of the aggregate model. 
        # The second is a dict with the metrics evaluated from the server side. In this case since we do not evaluate any global metric we simply return an empty dict
        return (
            Parameters(tensor_type="", tensors=[cast(bytes, global_model)]),
            {},
        )

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """
        Aggregate evaluation metrics using average.
        Note that this function do not compute a loss/metric for the central server but it only aggregate the loss/metrics of the various clients.
        """
        
        # Same comments of aggregate_fit
        if not results:
            return None, {}
        if not self.accept_failures and failures:
            return None, {}

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.evaluate_metrics_aggregation_fn:
            eval_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.evaluate_metrics_aggregation_fn(eval_metrics)
        elif server_round == 1:  # Only log this warning once
            flwr.common.logger.log(WARNING, "No evaluate_metrics_aggregation_fn provided")
        
        # The first value should contains the aggregations of the loss functions. 
        return 0, metrics_aggregated

    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """
        Evaluate model parameters using an evaluation function.
        This function compute a loss for the global model from the server side.
        """

        if self.evaluate_function is None:
            # No evaluation function provided
            return None

        eval_res = self.evaluate_function(server_round, parameters, {})
        if eval_res is None:
            return None

        loss, metrics = eval_res

        return loss, metrics

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

def aggregate(
    bst_prev_org: Optional[bytes],
    bst_curr_org: bytes,
) -> bytes:
    """
    Conduct bagging aggregation for given trees.
    Apparently this aggregation method is not defined in some paper but it's more a common way to do this.
    (Source: https://discuss.flower.ai/t/paper-on-bagging-aggregation-strategy-used-for-xgboost/114)

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    From another example I found this information :

    Bagging Aggregation
    Bagging (bootstrap) aggregation is an ensemble meta-algorithm in machine learning, used for enhancing the stability and accuracy of machine learning algorithms. Here, we leverage this algorithm for XGBoost trees.
    Specifically, each client is treated as a bootstrap by random subsampling (data partitioning in FL). At each FL round, all clients boost a number of trees (in this example, 1 tree) based on the local bootstrap samples. 
    Then, the clients' trees are aggregated on the server, and concatenates them to the global model from previous round. The aggregated tree ensemble is regarded as a new global model.
    This way, let's consider a scenario with M clients. Given FL round R, the bagging models consist of (M * R) trees.
    
    (Source : https://github.com/adap/flower/tree/main/examples/xgboost-comprehensive)
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

    When this function is called in aggregate_fit the first variable (bst_prev_org) is the global model and the second (bst_curr_org) is a client model
    """
    if not bst_prev_org:
        return bst_curr_org

    # Get the tree numbers
    tree_num_prev, _ = _get_tree_nums(bst_prev_org)
    _, paral_tree_num_curr = _get_tree_nums(bst_curr_org)
    
    # Convert them in json
    bst_prev = json.loads(bytearray(bst_prev_org))
    bst_curr = json.loads(bytearray(bst_curr_org))
    
    # Update tree numbers
    bst_prev["learner"]["gradient_booster"]["model"]["gbtree_model_param"][
        "num_trees"
    ] = str(tree_num_prev + paral_tree_num_curr)

    iteration_indptr = bst_prev["learner"]["gradient_booster"]["model"][
        "iteration_indptr"
    ]
    bst_prev["learner"]["gradient_booster"]["model"]["iteration_indptr"].append(
        iteration_indptr[-1] + paral_tree_num_curr
    )

    # Aggregate new trees
    trees_curr = bst_curr["learner"]["gradient_booster"]["model"]["trees"]
    for tree_count in range(paral_tree_num_curr):
        # Give tree a new ID
        trees_curr[tree_count]["id"] = tree_num_prev + tree_count
        
        # Add new tree
        bst_prev["learner"]["gradient_booster"]["model"]["trees"].append(
            trees_curr[tree_count]
        )

        bst_prev["learner"]["gradient_booster"]["model"]["tree_info"].append(0)

    bst_prev_bytes = bytes(json.dumps(bst_prev), "utf-8")

    return bst_prev_bytes


def _get_tree_nums(xgb_model_org: bytes) -> Tuple[int, int]:
    xgb_model = json.loads(bytearray(xgb_model_org))
    # Get the number of trees
    tree_num = int(
        xgb_model["learner"]["gradient_booster"]["model"]["gbtree_model_param"][
            "num_trees"
        ]
    )
    # Get the number of parallel trees
    paral_tree_num = int(
        xgb_model["learner"]["gradient_booster"]["model"]["gbtree_model_param"][
            "num_parallel_tree"
        ]
    )
    return tree_num, paral_tree_num

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

def evaluate_metrics_aggregation(eval_metrics):
    """Return an aggregated metric (AUC) for evaluation."""
    total_num = sum([num for num, _ in eval_metrics])
    auc_aggregated = (
        sum([metrics["AUC"] * num for num, metrics in eval_metrics]) / total_num
    )
    metrics_aggregated = {"AUC": auc_aggregated}
    return metrics_aggregated

def config_func(rnd: int) -> Dict[str, str]:
    """Return a configuration with global epochs."""
    config = {
        "global_round": str(rnd),
    }
    return config

def server_fn(context: Context):
    """
    Return a ServerApp COMPONENTS with the federated XGBoost Bagging strategy.
    The ServerAppComponents are, literally, the components required to create an istance of the server through the ServerApp function.

    ServerApp : https://flower.ai/docs/framework/ref-api/flwr.server.ServerApp.html
    ServerAppComponents : https://flower.ai/docs/framework/ref-api/flwr.server.ServerAppComponents.html

    Here more general info about App approach of flower :
    https://flower.ai/docs/framework/explanation-flower-architecture.html
    https://flower.ai/docs/framework/how-to-upgrade-to-flower-1.13.html
    """
    # Read from config
    num_rounds = context.run_config["num-server-rounds"]
    fraction_fit = context.run_config["fraction-fit"]
    fraction_evaluate = context.run_config["fraction-evaluate"]

    # Init an empty Parameter
    parameters = Parameters(tensor_type="", tensors=[])

    # Define strategy
    strategy = FedXgbBagging(
        fraction_fit=fraction_fit,
        fraction_evaluate=fraction_evaluate,
        evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation,
        on_evaluate_config_fn=config_func,
        on_fit_config_fn=config_func,
        initial_parameters=parameters,
    )
    config = ServerConfig(num_rounds = num_rounds)

    return ServerAppComponents(strategy = strategy, config = config)
