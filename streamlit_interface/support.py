"""
Support function used for the streamlit interface

@author: Alberto Zancanaro (Jesus)
@organization: Luxembourg Centre for Systems Biomedicine (LCSB)
@contact : alberto.zancanaro@uni.lu
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Imports

import pandas as pd
import pickle
import toml
import streamlit as st
import subprocess

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

def save_txt_list(filepath : str, data : list[str]) -> None:
    """
    Save a list of strings to a text file, each string on a new line.
    
    Parameters
    ----------
    filepath : str
        Path to the text file.
    data : list[str]
        List of strings to be saved.
    """

    with open(filepath, "w") as file:
        for line in data:
            file.write(f"{line}\n")

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def build_hist_computation_options() :
    hist_variable = read_txt_list("./streamlit_interface/field_hist.txt")

    bins_variable = st.selectbox(
        label = 'Select the variable for histogram computation',
        options = hist_variable, index = 0,
        key = 'bins_variable',
    )

    n_bins = st.slider(
        label = "N. of bins",
        min_value = 5, max_value = 15, step = 1,
        value = 10,
        key = 'n_bins',
    )

    st.write("Node to use for the computation")
    checkbox_node_1 = st.checkbox('Node 1', key = 'checkbox_node_1', value = True)
    checkbox_node_2 = st.checkbox('Node 2', key = 'checkbox_node_2', value = True)

    st.write("Other options")
    normalize_hist = st.checkbox('Normalize hist', key = 'normalize_hist', value = False)

    st.write("---")

    compute_hist_button = st.button(
        label = "Compute Histogram",
        key = "compute_hist_button",
        on_click = compute_hist,
    )

    config_dict = dict(
        bins_variable = bins_variable,
        n_bins = n_bins,
        normalize_hist = normalize_hist,
        checkbox_node_1 = checkbox_node_1,
        checkbox_node_2 = checkbox_node_2,
        histogram_computed = compute_hist_button
    )

    return config_dict

def build_hist_plot_options_matplotlib() :
    add_grid = st.checkbox('Display Grid', key = 'add_grid', value = True)
    add_edge = st.checkbox('Display edge', key = 'add_edge', value = True)
    alpha = st.slider("Alpha", min_value = 0., max_value = 1., step = 0.05)
    color = st.selectbox(
        label = "Select color",
        options = get_color_hex(), index = 0,
        format_func = get_color_name_from_hex,
        key = 'color'
    )

    plot_config_dict = dict(
        add_grid = add_grid,
        add_edge = add_edge,
        alpha = alpha,
        color = color,
    )

    return plot_config_dict

def build_hist_plot_options_streamlit() :
    color = st.selectbox(
        label = "Select color",
        options = get_color_hex(), index = 0,
        format_func = get_color_name_from_hex,
        key = 'color',
        on_change = draw_hist
    )

    plot_config_dict = dict(
        color = color
    )

    return plot_config_dict

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Histogram computation

def update_server_config() :
    # Get the server config from the streamlit interface
    server_config = dict(
        n_nodes = int(st.session_state.checkbox_node_1) + int(st.session_state.checkbox_node_2),
        max_number_of_attempts = 10,
        n_bins = st.session_state.n_bins,
        bins_variable = st.session_state.bins_variable,
        normalize_hist = st.session_state.normalize_hist,
        path_to_save = './results/',
    )

    # Save the server config to a toml file
    with open('./config/server_config.toml', 'w') as toml_file:
        toml.dump(server_config, toml_file)

def compute_hist() :
    update_server_config()

    subprocess.call(['sh', './other_scripts/run_hist_app.sh'])

    draw_hist()

def compute_hist_OLD() :
    from subprocess import Popen, PIPE, STDOUT
    p = Popen(['sh ./other_scripts/run_hist_app.sh'], stdout = PIPE,
            stderr = STDOUT, shell = True)
    for line in p.stdout:
        st.write(line)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Histogram plot

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

def load_data_for_plotting() :
    
    # Load the results
    with open('./results/results.pkl', 'rb') as f :
        results = pickle.load(f)
    
    # Compute the labels for the bins
    x_labels = []
    bins = results['bins']
    for i in range(len(bins) - 1) : x_labels.append(f"{round(bins[i], 1)}-{round(bins[i + 1], 1)}")
    results['labels'] = x_labels

    return results

def draw_hist() :

    results = load_data_for_plotting()

    x_label = results['bins_variable']

    if min(results['histogram']) >= 0 and max(results['histogram']) <= 1 :
        y_label = 'Proportion of samples'
    else :
        y_label = 'Number of samples'

    results = pd.DataFrame({
        "labels" : results['labels'],
        "histogram" : results['histogram'],
    })

    st.bar_chart(
        data = results,
        x = 'labels', y = 'histogram',
        x_label = x_label, y_label = y_label,
        color = st.session_state.color,
    )


