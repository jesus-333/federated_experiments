"""

@author: Alberto Zancanaro (Jesus)
@organization: Luxembourg Centre for Systems Biomedicine (LCSB)
@contact : alberto.zancanaro@uni.lu
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Imports

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import streamlit as st

from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import support_ml_app
import support_interface_hist

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def plot_decision_boundary(streamlit_container_for_the_plot) :
    path_data = "./data/server_data.csv"
    
    # Load data 
    data, labels = load_data_for_plotting(path_data)

    # Get dimensionality reduction method and reduce data dimensionality
    dimen_reduction_method = get_dimesionality_reducttion_method()
    data_2D = reduce_data_dimensionality(data, dimen_reduction_method)
    
    # Used to model inizialiation
    n_classes = 3
    n_features = data.shape[1]

    # Get the model
    model = get_model(n_classes, n_features)

    # Create a meshgrid
    x_min, x_max = data_2D[:, 0].min(), data_2D[:, 0].max()
    y_min, y_max = data_2D[:, 1].min(), data_2D[:, 1].max()
    diff_x = np.abs(np.diff(data_2D[:, 0]))
    diff_y = np.abs(np.diff(data_2D[:, 1]))
    step_x = np.mean(diff_x)
    step_y = np.mean(diff_y)
    print(x_min, x_max, y_min, y_max)
    print(diff_x, diff_y)
    print(step_x, step_y)
    print(data_2D[:, 0], data_2D.shape)
    xx, yy = get_meshgrid(x_min, x_max, y_min, y_max, step_x = step_x, step_y = step_y, padding_percentage = st.session_state.grid_padding)

    # Predict the class for each point in the meshgrid
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    # grid_points_original_space = reverse_data_dimensionality(grid_points, dimen_reduction_method)
    #
    # z = model.predict(grid_points_original_space)
    # z = z.reshape(xx.shape)
    
    fig, ax = get_scatter_data_plot(data_2D, labels)

    # Plot the decision boundary
    # ax.contourf(xx, yy, z, alpha = 0.3, cmap = 'coolwarm')
    ax.set_title(f"Decision boundary of {st.session_state.ml_model_name} model")

    with streamlit_container_for_the_plot :
        st.pyplot(fig)

def get_scatter_data_plot(data_2D : np.ndarray, labels : np.ndarray) :
    fig, ax = plt.subplots(figsize = (12, 4))

    for class_label in np.unique(labels) :
        idx = labels == class_label
        ax.scatter(data_2D[idx, 0], data_2D[idx, 1], 
                   label = f'Class {class_label}', alpha = 0.7,
                   color = st.session_state[f'color_class_{class_label}'],
                   edgecolors = 'k',
                   )

    if st.session_state.dimensionality_reduction == 'none' :
        ax.set_xlabel(f'component 1: {st.session_state.clf_variable_1}')
        ax.set_ylabel(f'component 2: {st.session_state.clf_variable_2}')
    else :
        ax.set_xlabel('component 1')
        ax.set_ylabel('component 2')
    
    return fig, ax

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Data function

def get_dimesionality_reducttion_method() :
    if st.session_state.dimensionality_reduction == 'None' :
        return None
    elif st.session_state.dimensionality_reduction == 'PCA' :
        dim_reduction_method = PCA(n_components = 2)
    elif st.session_state.dimensionality_reduction == 't-SNE' :
        dim_reduction_method = TSNE(n_components = 2, init = 'random')
    else :
        raise ValueError(f"Dimensionality reduction method '{st.session_state.dimensionality_reduction}' not recognized.")

    return dim_reduction_method

def reduce_data_dimensionality(data : np.ndarray, dim_reduction_method) -> np.ndarray :
    if st.session_state.dimensionality_reduction == 'None' :
        # Get the two variable to plot
        clf_variables_list = np.asarray(support_interface_hist.read_txt_list("./streamlit_interface/field_hist.txt"))
        clf_variable_1 = st.session_state.clf_variable_1
        clf_variable_2 = st.session_state.clf_variable_2
        
        # Get the idx
        idx_1 = clf_variables_list == clf_variable_1
        idx_2 = clf_variables_list == clf_variable_2
        
        # Get the two dimension
        data_2D = np.asarray([data[:, idx_1], data[:, idx_2]])

        # Remove the extra dimension and transpose
        data_2D = np.squeeze(data_2D).T
    else :
        data_2D = dim_reduction_method.fit_transform(data)

    return data_2D

def reverse_data_dimensionality(data_2D : np.ndarray, dim_reduction_method) -> np.ndarray :
    if st.session_state.dimensionality_reduction == 'None' :
        pass
    elif st.session_state.dimensionality_reduction == 'PCA' :
        reversed_data = dim_reduction_method.inverse_transform(data_2D)
    elif st.session_state.dimensionality_reduction == 't-SNE' :
        raise NotImplementedError("t-SNE cannot be inverted.")
    else :
        raise ValueError(f"Dimensionality reduction method '{st.session_state.dimensionality_reduction}' not recognized.")

    return reversed_data

@st.cache_data
def get_meshgrid(x_min, x_max, y_min, y_max, step_x :float, step_y : float, padding_percentage = 0.05) :
    """
    Create a meshgrid for plotting decision boundaries.
    """
    
    # Check inputs
    if x_min == x_max : raise ValueError(f"x_min and x_max cannot be the same. Got {x_min}.")
    if y_min == y_max : raise ValueError(f"y_min and y_max cannot be the same. Got {y_min}.")
    if padding_percentage <= 0 or padding_percentage > 1 : raise ValueError(f"Padding percentage must be between 0 and 1. Got {padding_percentage}.")
    if step_x <= 0 : raise ValueError(f"step_x must be positive. Got {step_x}.")
    if step_y <= 0 : raise ValueError(f"step_y must be positive. Got {step_y}.")

    # Padding, to avoid points on the edge of the plot
    x_min = x_min * (1 + padding_percentage) if x_min < 0 else x_min * (1 - padding_percentage)
    x_max = x_max * (1 - padding_percentage) if x_max < 0 else x_max * (1 + padding_percentage)
    y_min = y_min * (1 + padding_percentage) if y_min < 0 else y_min * (1 - padding_percentage)
    y_max = y_max * (1 - padding_percentage) if y_max < 0 else y_max * (1 + padding_percentage)

    # Handle the case where min and max are 0 (basically it set the values as a percentage of the values range)
    if x_min == 0 : x_min = x_max * padding_percentage
    if x_max == 0 : x_max = abs(x_min) * padding_percentage # Note that if x_max is 0 then x_min is negative
    if y_min == 0 : y_min = y_max * padding_percentage
    if y_max == 0 : y_max = abs(y_min) * padding_percentage # Note that if y_max is 0 then y_min is negative

    # Create the meshgrid
    num_x = int(np.ceil((x_max - x_min) / step_x))
    num_y = int(np.ceil((y_max - y_min) / step_y))
    print(num_x, num_y)
    xx, yy = np.meshgrid(np.arange(x_min, x_max, num_x), np.arange(y_min, y_max, num_y))

    return xx, yy

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Model function

def get_model(n_classes, n_features) :
    """
    Create the ML model and load the parameters from a pkl file.
    The model is created according to settings obtained from the streamlit session state.

    """
    ml_model_name = st.session_state.ml_model_name

    type_of_params = st.session_state.type_of_params

    if type_of_params == 'FL training' :
        path_ml_model_params = f'./results/final_params_{ml_model_name}.pkl'
    elif type_of_params == 'Only client 1' :
        path_ml_model_params = f'./results/trained_params_{ml_model_name}_node_0.pkl'
    elif type_of_params == 'Only client 2' :
        path_ml_model_params = f'./results/trained_params_{ml_model_name}_node_1.pkl'

    model = create_model_and_load_params(ml_model_name, path_ml_model_params, n_classes, n_features)

    return model

@st.cache_data
def create_model_and_load_params(ml_model_name : str, path_ml_model_params : str, n_classes : int, n_features : int) -> np.ndarray :
    # Create the ml model
    model = create_model(ml_model_name)

    # Load the params from a pkl file
    with open(path_ml_model_params, 'rb') as f : params = pickle.load(f)

    # Load the parameters into the model
    support_ml_app.set_initial_params(ml_model_name, model, n_classes, n_features) # This is required to initialize the params attributes inside the model object
    support_ml_app.set_model_params(ml_model_name, model, params)

    return model

@st.cache_data
def create_model(ml_model_name : str) :
    if ml_model_name == 'SVM' :
        model = svm.LinearSVC()

    return model

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

@st.cache_data
def load_data_for_plotting(path_data : str):
    data_fields = support_ml_app.read_txt_list("./config/fields_ml.txt")

    # Load dataset
    data, labels, _ = support_ml_app.get_data(path_data, data_fields)

    return data, labels

@st.cache_data
def load_model_for_plotting(path_data : str) :
    pass

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def get_color_hex() :
    color_hex = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    return color_hex

def get_color_name_from_hex(hex : str) :
    hex_to_name = {
        '#1f77b4' : 'blue',
        '#ff7f0e' : 'orange',
        '#2ca02c' : 'green',
        '#d62728' : 'red',
        '#9467bd' : 'purple',
        '#8c564b' : 'brown',
        '#e377c2' : 'pink',
        '#7f7f7f' : 'gray',
        '#bcbd22' : 'olive',
        '#17becf' : 'cyan',
    }

    return hex_to_name[hex]
