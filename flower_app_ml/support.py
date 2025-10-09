"""
Support function for the ml training

@author: Alberto Zancanaro (Jesus)
@organization: Luxembourg Centre for Systems Biomedicine (LCSB)
@contact : alberto.zancanaro@uni.lu
@date: September 2025
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Imports

import numpy as np
import sklearn.svm as svm

from flwr.common import NDArrays

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def get_ml_model(ml_model_name : str, ml_model_config : dict) :
    if ml_model_name == 'SVC' :
        model = svm.SVC(kernel = ml_model_config['kernel'], max_iter = ml_model_config['max_iter'])
    else :
        raise ValueError(f"ML algorithm {ml_model_name} not implemented")

    return model

def get_model_parameters(ml_model_name : str, ml_model) -> NDArrays:
    """
    Return the parameters of a sklearn model
    """
    if ml_model_name == 'SVC' :
        params = [
            ml_model.coef_,
            ml_model.intercept_,
        ]
    else:
        raise ValueError(f"ML algorithm {ml_model_name} not implemented")

    return params

def set_model_params(ml_model_name : str, ml_model, params: NDArrays) :
    """
    Set the parameters of a sklean model
    """

    if ml_model_name == 'SVC' :
        ml_model.coef_ = params[0]
        if ml_model.fit_intercept:
            ml_model.intercept_ = params[1]

    else:
        raise ValueError(f"ML algorithm {ml_model_name} not implemented")
    return ml_model


def set_initial_params(ml_model_name : str, ml_model, n_classes: int, n_features: int) :
    """
    Set initial parameters as zeros.
    
    From a Flower tutorial (https://github.com/adap/flower/blob/main/examples/quickstart-sklearn-tabular/sklearnexample/task.py)
    Required since model params are uninitialized until model.fit is called but server
    asks for initial parameters from clients at launch. Refer to
    sklearn.linear_model.LogisticRegression documentation for more information.
    """
    
    if ml_model_name == 'SVC' :
        # Setup the model classes
        ml_model.classes_ = np.array([i for i in range(n_classes)])

        coef = np.zeros((n_classes, n_features))
        intercept = np.zeros((n_classes,))

        initial_param = [coef, intercept]
    elif ml_model_name == 'k-means' :
        initial_param = get_kmeans_initial_parameters()
    else:
        raise ValueError(f"ML algorithm {ml_model_name} not implemented")

    ml_model = set_model_params(ml_model_name, ml_model, initial_param)
    return ml_model

def get_kmeans_initial_parameters() :
    pass

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def read_txt_list(filepath : str) -> list[str]:
    """
    Read a text file and return its content as a list of strings.
    
    Parameters
    ----------
    filepath : str
        Path to the text file.

    Returns
    -------
    list[str]
        List of strings, each representing a line in the text file.
    """

    with open(filepath, "r") as file:
        lines = file.readlines()
    
    # Remove whitespace characters like `\n` at the end of each line
    lines = [line.strip() for line in lines if line.strip()]
    
    return lines
