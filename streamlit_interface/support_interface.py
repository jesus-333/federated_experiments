"""
Support function used for the streamlit interface

@author: Alberto Zancanaro (Jesus)
@organization: Luxembourg Centre for Systems Biomedicine (LCSB)
@contact : alberto.zancanaro@uni.lu
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Imports

import toml
import streamlit as st
import subprocess

import support_plot

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

    column_proportions = [0.5, 0.5]
    node_option_column, bins_options_column = st.columns(column_proportions)

    with bins_options_column :
        st.radio(
            label = "Bins Distribution",
            options = ["Uniform", "Logarithmic"],
            captions = [
                "Bins are evenly distributed between min and max",
                "bins are logarithmically distributed between min and max"
            ],
            key = 'bins_distribution',
        )

    with node_option_column :
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

def build_hist_plot_options(streamlit_container_for_the_plot) :
    plot_backend = st.selectbox(
        label = 'Select the plot backend',
        options = ['matplotlib', 'streamlit'], index = 0,
        key = 'plot_backend',
        on_change = support_plot.draw_hist,
        args = [streamlit_container_for_the_plot]
    )

    st.radio(
        label = "Plot type",
        options = ["Type 1", "Type 2", "Type 3"],
        captions = [
            "1 plot, all class mixed",
            "3, plot, classes separated",
            "1 plot, classes separated"],
        key = 'plot_type',
        on_change = support_plot.draw_hist,
        args = [streamlit_container_for_the_plot]
    )

    st.write("Other options")
    st.checkbox(
        label = 'Normalize hist',
        key = 'normalize_hist',
        value = False,
        on_change = support_plot.draw_hist,
        args = [streamlit_container_for_the_plot]
    )

    st.selectbox(
        label = "Select color",
        options = support_plot.get_color_hex(), index = 0,
        format_func = support_plot.get_color_name_from_hex,
        key = 'color',
        on_change = support_plot.draw_hist,
        args = [streamlit_container_for_the_plot]
    )

    if plot_backend == 'matplotlib' :
        build_hist_plot_options_matplotlib(streamlit_container_for_the_plot)
    elif plot_backend == 'streamlit' :
        pass

def build_hist_plot_options_matplotlib(streamlit_container_for_the_plot) :

    st.slider(label = "Alpha", key = 'alpha',
              min_value = 0.5, max_value = 1., value = 1., step = 0.05,
              on_change = support_plot.draw_hist_matplotlib, args = [streamlit_container_for_the_plot]
              )

    checkbox_col_1, checkbox_col_2 = st.columns([0.5, 0.5])

    with checkbox_col_1 :
        st.checkbox('Display Grid', key = 'add_grid', value = True, on_change = support_plot.draw_hist_matplotlib, args = [streamlit_container_for_the_plot])
        st.checkbox('Display edge', key = 'add_edge', value = True, on_change = support_plot.draw_hist_matplotlib, args = [streamlit_container_for_the_plot])

    with checkbox_col_2 :
        st.checkbox(
            label = 'Display Mean', key = 'add_mean', value = False,
            on_change = support_plot.draw_hist_matplotlib, args = [streamlit_container_for_the_plot]
        )

        st.checkbox(
            label = 'Display Std', key = 'add_std', value = False,
            on_change = support_plot.draw_hist_matplotlib, args = [streamlit_container_for_the_plot]
        )

        st.checkbox(
            label = 'Log scale y axis', key = 'y_axis_log', value = False,
            on_change = support_plot.draw_hist, args = [streamlit_container_for_the_plot]
        )


    st.button(
        label = "Save Histogram",
        key = "save_hist_button",
    )

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Histogram computation

def update_server_config() :
    # Get the server config from the streamlit interface
    server_config = dict(
        n_nodes = int(st.session_state.checkbox_node_1) + int(st.session_state.checkbox_node_2),
        max_number_of_attempts = 10,
        n_bins = st.session_state.n_bins,
        bins_variable = st.session_state.bins_variable,
        bins_distribution = st.session_state.bins_distribution.lower(),
        path_to_save = './results/',
    )

    # Save the server config to a toml file
    with open('./config/server_config.toml', 'w') as toml_file:
        toml.dump(server_config, toml_file)

def compute_hist(streamlit_container) :
    update_server_config()

    subprocess.call(['sh', './other_scripts/run_hist_app.sh'])

    if st.session_state.plot_backend == 'matplotlib' :
        support_plot.draw_hist_matplotlib(streamlit_container)
    elif st.session_state.plot_backend == 'streamlit' :
        support_plot.draw_hist_streamlit(streamlit_container)

def compute_hist_OLD() :
    from subprocess import Popen, PIPE, STDOUT
    p = Popen(['sh ./other_scripts/run_hist_app.sh'], stdout = PIPE,
            stderr = STDOUT, shell = True)
    for line in p.stdout:
        st.write(line)
