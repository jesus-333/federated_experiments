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

import pickle
import warnings

from flwr.client import ClientApp
from flwr.common import ArrayRecord, Context, Message, MetricRecord, RecordDict

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

    # Get node config
    path_client_data = context.node_config["path_client_data"]
    node_id = context.node_config['node_id']

    # Get ml model config
    ml_model_config = msg.content["config"]
    ml_model_name = ml_model_config['ml_model_name']

    # Load the data
    x_train, y_train, _ = support.get_data(path_client_data, ml_model_config['fields_to_use_for_the_train'])

    # Create ml model
    ml_model = support.get_ml_model(ml_model_name, ml_model_config)

    # Setting initial parameters
    # Required because the model parameters are not initialized until the fit function is called
    support.set_initial_params(ml_model_name, ml_model, 3, x_train.shape[1])

    # Apply received parameters
    params = msg.content["arrays"].to_numpy_ndarrays()
    support.set_model_params(ml_model_name, ml_model, params)

    # Ignore convergence failure due to low local epochs
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Train the model on local data
        ml_model.fit(x_train, y_train)

    # Let's compute train loss
    # y_train_pred_proba = ml_model.predict_proba(x_train)
    # train_logloss = log_loss(y_train, y_train_pred_proba)

    # Extract the trained model parameters
    params_trained = support.get_model_params(ml_model_name, ml_model)
    model_record = ArrayRecord(params_trained)

    # Prepare metrics
    metrics = {"num-examples": len(x_train)}
    metric_record = MetricRecord(metrics)

    # Construct a Message with the results
    content = RecordDict({"arrays": model_record, "metrics": metric_record})

    # Save the model weights locally
    path_to_save_model = context.node_config['path_to_save_model'] if 'path_to_save_model' in context.node_config else './'
    with open(f'{path_to_save_model}/trained_params_{ml_model_name}_node_{node_id}.pkl', "wb") as f : pickle.dump(params, f)

    return Message(content = content, reply_to = msg)

@app.query()
def query(msg : Message, context : Context) :
    """
    Send the weights of the trained model to the server.

    Implemented to obtain the model weights of a single nodes outside the FL process. From what I see the FedAvg strategy only return the finale model weights after the last round.
    For the PoC I also need the model weights of each node to show the difference between the models trained on different datasets.
    """
    
    # Get config
    node_id = context.node_config['node_id']
    path_to_save_model = context.node_config['path_to_save_model'] if 'path_to_save_model' in context.node_config else './'
    my_config = msg.content['my_config']

    # Load the model weights
    ml_model_name = my_config['ml_model_name']
    with open(f"{path_to_save_model}/trained_params_{ml_model_name}_node_{node_id}.pkl", "rb") as f : params = pickle.load(f)
    
    model_weights = dict()
    if my_config['ml_model_name'] == 'SVM' :
        # For SVM the params are [coef, intercept], coef is of shape (n_classes, n_features)
        
        for i in range(len(params[0])) :
            model_weights[f'coef_class_{i}'] = list(params[0][i])

        model_weights['intercept'] = list(params[1])

    print(model_weights)

    # Prepare the Message to send the model weights to the server
    model_record = MetricRecord(model_weights)
    reply_content = RecordDict({"query_results": model_record})
    # reply_content = RecordDict({"query_results": MetricRecord(query_results)})

    return Message(content = reply_content, reply_to = msg)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

