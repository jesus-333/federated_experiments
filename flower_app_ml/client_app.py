"""
A Flower `ClientApp` that train a machine learning algorithm
Currently implemented algorithm are LDA, SVM, neural network, k-means.

@author: Alberto Zancanaro (Jesus)
@organization: Luxembourg Centre for Systems Biomedicine (LCSB)
@contact : alberto.zancanaro@uni.lu
@date: September 2025
"""
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Imports

import numpy as np
import pandas as pd
import warnings

from collections.abc import Iterable

from flwr.client import ClientApp
from flwr.common import Context, Message, MetricRecord, RecordDict

import support

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# warnings.filterwarnings("ignore", category = UserWarning)

# Flower ClientApp
app = ClientApp()

@app.train()
def train(msg: Message, context: Context):
    """
    Train the model on local data.
    """
    
    # Get config
    my_config = msg.content["config"]
    ml_model_name = my_config['ml_model_name']
    ml_model_config = my_config['ml_model_config']

    # Create ml model
    ml_model = support.get_ml_model(ml_model_name, ml_model_config)

    # Setting initial parameters
    # Required because the model parameters are not initialized until the fit function is called
    support.set_initial_params(my_config['ml_model_name'], ml_model)

    # Apply received pararameters
    params = msg.content["arrays"].to_numpy_ndarrays()
    support.set_model_params(ml_model_name, ml_model_name, params)

    # Load the data
    x_train, y_train = get_data()

    # Ignore convergence failure due to low local epochs
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Train the model on local data
        ml_model.fit(x_train, y_train)

    # Let's compute train loss
    y_train_pred_proba = model.predict_proba(X_train)
    train_logloss = log_loss(y_train, y_train_pred_proba)

    # Construct and return reply Message
    ndarrays = get_model_params(model)
    model_record = ArrayRecord(ndarrays)
    metrics = {"num-examples": len(X_train), "train_logloss": train_logloss}
    metric_record = MetricRecord(metrics)
    content = RecordDict({"arrays": model_record, "metrics": metric_record})
    return Message(content=content, reply_to=msg)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def get_data(path_client_data : str, bins_variable : str) -> tuple[np.ndarray, np.ndarray]:
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

    name_numerical_features = read_txt_list(f'{path_client_data}fields_clf.txt')
    
    # Get the labels (string format)
    labels_str = dataset_client['Diagnosis'].to_numpy()

    # Convert the labels to integers
    labels_str_to_int = {'control': 0, 'UC': 1, 'CD': 2}
    
    # Get the labels (integer format)
    label_int = [labels_str_to_int[label] for label in labels_str]


    return x_data, y_data
