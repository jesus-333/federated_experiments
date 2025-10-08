"""
Support function used for the streamlit interface

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

def build_hist_computation_options(streamlit_container) :
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

    compute_hist_button = st.button(
        label    = "Compute Histogram",
        key      = "compute_hist_button",
        on_click = compute_hist,
        args = [streamlit_container]
    )

    config_dict = dict(
        bins_variable = bins_variable,
        n_bins = n_bins,
        checkbox_node_1 = checkbox_node_1,
        checkbox_node_2 = checkbox_node_2,
        histogram_computed = compute_hist_button
    )

    return config_dict

def build_hist_plot_options_matplotlib(streamlit_container) :
    color = st.selectbox(
        label = "Select color",
        options = get_color_hex(), index = 0,
        format_func = get_color_name_from_hex,
        key = 'color',
        on_change = draw_hist,
        args = [streamlit_container]
    )

    checkbox_col_1, checkbox_col_2 = st.columns([0.5, 0.5])

    with checkbox_col_1 :
        add_grid = st.checkbox('Display Grid', key = 'add_grid', value = True, on_change = draw_hist_matplotlib, args = [streamlit_container])
        add_edge = st.checkbox('Display edge', key = 'add_edge', value = True, on_change = draw_hist_matplotlib, args = [streamlit_container])

    with checkbox_col_2 :
        add_mean = st.checkbox('Display Mean', key = 'add_mean', value = False, on_change = draw_hist_matplotlib, args = [streamlit_container])
        add_std = st.checkbox('Display Std', key = 'add_std', value = False, on_change = draw_hist_matplotlib, args = [streamlit_container])

    alpha = st.slider("Alpha", min_value = 0.5, max_value = 1., value = 1., step = 0.05, key = 'alpha', on_change = draw_hist_matplotlib, args = [streamlit_container])

    save_hist_button = st.button(
        label = "Save Histogram",
        key = "save_hist_button",
    )

    plot_config_dict = dict(
        add_grid = add_grid,
        add_edge = add_edge,
        add_mean = add_mean,
        add_std = add_std,
        alpha = alpha,
        color = color,
    )

    return plot_config_dict

def build_hist_plot_options_streamlit(streamlit_container) :
    color = st.selectbox(
        label = "Select color",
        options = get_color_hex(), index = 0,
        format_func = get_color_name_from_hex,
        key = 'color',
        on_change = draw_hist_streamlit,
        args= [streamlit_container]
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
        path_to_save = './results/',
    )

    # Save the server config to a toml file
    with open('./config/server_config.toml', 'w') as toml_file:
        toml.dump(server_config, toml_file)

def compute_hist(streamlit_container) :
    update_server_config()

    subprocess.call(['sh', './other_scripts/run_hist_app.sh'])

    if st.session_state.plot_backend == 'matplotlib' :
        draw_hist_matplotlib(streamlit_container)
    elif st.session_state.plot_backend == 'streamlit' :
        draw_hist_streamlit(streamlit_container)

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

def get_matplotlib_config() -> dict :
    """
    Get the matplotlib configuration from the streamlit session state.
    If a configuration option is not set in the session state, a default value is used.

    This function was created to avoid buggy behavior of streamlit when using.
    It basically protects againsst the cases when one element of the UI is not created yet but its value is needed in the plotting function.
    """

    matplotlib_config = dict()

    if 'color' not in st.session_state : matplotlib_config['color'] = '#1f77b4'
    else : matplotlib_config['color'] = st.session_state.color

    if 'add_grid' not in st.session_state : matplotlib_config['add_grid'] = True
    else : matplotlib_config['add_grid'] = st.session_state.add_grid

    if 'add_edge' not in st.session_state : matplotlib_config['add_edge'] = True
    else : matplotlib_config['add_edge'] = st.session_state.add_edge

    if 'add_mean' not in st.session_state : matplotlib_config['add_mean'] = False
    else : matplotlib_config['add_mean'] = st.session_state.add_mean

    if 'add_std' not in st.session_state : matplotlib_config['add_std'] = False
    else : matplotlib_config['add_std'] = st.session_state.add_std

    if 'alpha' not in st.session_state : matplotlib_config['alpha'] = 1.
    else : matplotlib_config['alpha'] = st.session_state.alpha

    return matplotlib_config

def create_hist_matplotlib(results : dict) :
    
    # Get data from results dict
    labels = results['labels']
    bins = np.asarray(results['bins'])
    histogram = results['histogram']
    
    # Compute width
    width = (bins[1] - bins[0])

    # Get matplotlib config
    matplotlib_config = get_matplotlib_config()

    # Create the plot
    fig, ax = plt.subplots(figsize = (12, 6))

    # Plot histogram
    ax.bar(bins[:-1], histogram,
        width = width, align = 'edge',
        edgecolor = 'black' if matplotlib_config['add_edge'] else None,
        color = matplotlib_config['color'], alpha = matplotlib_config['alpha']
    )

    # (OPTIONAL) Add mean line
    if matplotlib_config['add_mean'] :
        ax.axvline(results['mean'], color = 'red', linestyle = 'dashed', linewidth = 1)
        ax.text(results['mean'] * 1.05, max(histogram) * 0.9, f'Mean: {results["mean"]:.2f}', color = 'red')

    # (OPTIONAL) Add std lines as shaded area
    if matplotlib_config['add_std'] :
        ax.axvline(results['mean'] - results['std'], color = 'orange', linestyle = 'dashed', linewidth = 1)
        ax.axvline(results['mean'] + results['std'], color = 'orange', linestyle = 'dashed', linewidth = 1)
        ax.fill_betweenx([0, max(histogram) * 1.2], results['mean'] - results['std'], results['mean'] + results['std'], color = 'orange', alpha = 0.2)
        ax.text((results['mean'] + results['std']) * 1.02, max(histogram) * 0.9, f'Std: {results["std"]:.2f}', color = 'orange')

    # Add xticks
    ax.set_xticks(bins)

    # Add title and axis labels
    ax.set_title(f'Mean: {results['mean']:.2f}, Std: {results['std']:.2f}, N: {results['n_samples']}')
    ax.set_xlabel(results['bins_variable'])
    if min(histogram) >= 0 and max(histogram) <= 1 :
        ax.set_ylabel('Proportion of samples')
    else :
        ax.set_ylabel('Number of samples')

    # Add Grid
    if matplotlib_config['add_grid'] :
        ax.set_axisbelow(True)
        ax.grid(True)

    # Adjust limits
    # ax.set_xlim([min(bins) * 0.95, max(bins) * 1.02])
    ax.set_ylim([0, max(histogram) * 1.1])


    fig.tight_layout()

    return fig, ax

def draw_hist(streamlit_container : st.delta_generator.DeltaGenerator) :
    if st.session_state.plot_backend == 'matplotlib' :
        draw_hist_matplotlib(streamlit_container)
    elif st.session_state.plot_backend == 'streamlit' :
        draw_hist_streamlit(streamlit_container)

def draw_hist_matplotlib(streamlit_container) :
    # Load the data
    results = load_data_for_plotting()
    
    # Create the figure
    fig, ax = create_hist_matplotlib(results)

    # Draw the figure
    with streamlit_container :
        st.pyplot(fig)
    

def draw_hist_streamlit(streamlit_container) :

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

    with streamlit_container :

        st.bar_chart(
            data = results,
            x = 'labels', y = 'histogram',
            x_label = x_label, y_label = y_label,
            color = st.session_state.color,
        )
