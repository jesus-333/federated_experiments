"""
Function used to draw the histogram

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

from collections.abc import Iterable

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Main draw function

def draw_hist(streamlit_container : st.delta_generator.DeltaGenerator) :
    if st.session_state.plot_backend == 'matplotlib' :
        draw_hist_matplotlib(streamlit_container)
    elif st.session_state.plot_backend == 'streamlit' :
        draw_hist_streamlit(streamlit_container)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Matplotlib backend

def draw_hist_matplotlib(streamlit_container) :
    # Load the data
    results = load_data_for_plotting()

    # Get matplotlib config
    matplotlib_config = get_matplotlib_config()
    
    # Create the figure
    fig, ax = create_hist_matplotlib(results, matplotlib_config)

    # Draw the figure
    with streamlit_container :
        st.pyplot(fig)


def get_matplotlib_config() -> dict :
    """
    Get the matplotlib configuration from the streamlit session state.
    If a configuration option is not set in the session state, a default value is used.

    This function was created to avoid buggy behavior of streamlit when using.
    It basically protects against the cases when one element of the UI is not created yet but its value is needed in the plotting function.
    """

    matplotlib_config = dict()

    if 'bins_distribution' not in st.session_state : matplotlib_config['bins_distribution'] = 'uniform'
    else : matplotlib_config['bins_distribution'] = st.session_state.bins_distribution.lower()

    if 'plot_type' not in st.session_state : matplotlib_config['plot_type'] = "Type 1"
    else : matplotlib_config['plot_type'] = st.session_state.plot_type
    
    if 'normalize_hist' not in st.session_state : matplotlib_config['normalize_hist'] = False
    else : matplotlib_config['normalize_hist'] = st.session_state.normalize_hist

    if 'color' not in st.session_state : matplotlib_config['color'] = '#1f77b4'
    else : matplotlib_config['color'] = st.session_state.color

    if 'alpha' not in st.session_state : matplotlib_config['alpha'] = 1.
    else : matplotlib_config['alpha'] = st.session_state.alpha

    if 'add_grid' not in st.session_state : matplotlib_config['add_grid'] = True
    else : matplotlib_config['add_grid'] = st.session_state.add_grid

    if 'add_edge' not in st.session_state : matplotlib_config['add_edge'] = True
    else : matplotlib_config['add_edge'] = st.session_state.add_edge

    if 'add_mean' not in st.session_state : matplotlib_config['add_mean'] = False
    else : matplotlib_config['add_mean'] = st.session_state.add_mean

    if 'add_std' not in st.session_state : matplotlib_config['add_std'] = False
    else : matplotlib_config['add_std'] = st.session_state.add_std

    if 'y_axis_log' not in st.session_state : matplotlib_config['y_axis_log'] = False
    else : matplotlib_config['y_axis_log'] = st.session_state.y_axis_log

    return matplotlib_config

def create_hist_matplotlib(results : dict, matplotlib_config : dict) :
    # Get data from results dict
    # labels = results['labels']

    # Create the plot
    fig, ax = plt.subplots(figsize = (18, 5))

    # Plot histogram
    if matplotlib_config['plot_type'] == 'Type 1' :
        histogram = results['histogram']
        if matplotlib_config['normalize_hist'] : histogram = histogram / np.sum(histogram)
        bins = np.asarray(results['bins'])
        
        # Plot the data
        ax = plot_data_inside_ax(ax, bins, histogram, matplotlib_config)

        # Beautify the plot
        fig, ax = beautify_hist_matplotlib(fig, ax, results, histogram, bins, matplotlib_config)
    elif matplotlib_config['plot_type'] == 'Type 2' :
        fig, axs = plt.subplots(1, 3, figsize = (18, 5))
    
        class_list = ['UC', 'CD', 'control']
        for i in range(len(class_list)) :
            class_to_plot = class_list[i]

            # Get the data for the current class
            results_for_the_class = results[f'results_{class_to_plot}']
            histogram = results_for_the_class['histogram']
            if matplotlib_config['normalize_hist'] : histogram = histogram / np.sum(histogram)
            bins = np.asarray(results_for_the_class['bins'])

            # Plot the data
            axs[i] = plot_data_inside_ax(axs[i], bins, histogram, matplotlib_config, label = class_to_plot)
            axs[i].set_title(f'{class_to_plot}')

            # Beautify the plot
            fig, axs[i] = beautify_hist_matplotlib(fig, axs[i], results_for_the_class, histogram, bins, matplotlib_config)

    elif matplotlib_config['plot_type'] == 'Type 3' :
        fig, ax = plt.subplots(1, 1, figsize = (18, 5))
        del matplotlib_config['color'] # Remove color from the config to use default matplotlib color cycle
    
        class_list = ['UC', 'CD', 'control']
        total_bottom = np.zeros(len(results['results_UC']['histogram']))
        for i in range(len(class_list)) :
            class_to_plot = class_list[i]

            # Get the data for the current class
            results_for_the_class = results[f'results_{class_to_plot}']
            histogram = results_for_the_class['histogram']
            if matplotlib_config['normalize_hist'] : histogram = histogram / np.sum(results['results_all']['histogram']) # Normalize by the total population
            bins = np.asarray(results_for_the_class['bins'])

            # Plot the data
            ax = plot_data_inside_ax(ax, bins, histogram, matplotlib_config, label = class_to_plot, bottom = total_bottom)

            total_bottom += histogram

        # Beautify the plot
        # Note that histogram is passed to the function to get the y-axis limits correctly.
        # Since we are stacking the histograms, the maximum value of the last histogram is not the maximum value of the stacked histogram.
        # Therefore, we pass the total_bottom to the function to get the correct y-axis limits (which is also the histogram of the total population).
        fig, ax = beautify_hist_matplotlib(fig, ax, results['results_all'], total_bottom, bins, matplotlib_config)
        ax.legend(title = 'Classes')

    return fig, ax

def plot_data_inside_ax(ax : plt.Axes, bins : np.ndarray, histogram : np.ndarray, matplotlib_config : dict, label : str = None, bottom = None) -> plt.Axes :
    # Compute width
    width = np.diff(bins)

    # Plot hist
    ax.bar(bins[:-1], histogram,
           width = width, align = 'edge',
           edgecolor = 'black' if matplotlib_config['add_edge'] else None,
           color = matplotlib_config['color'] if 'color' in matplotlib_config else None, alpha = matplotlib_config['alpha'],
           label = label, bottom = bottom
           )
    return ax

def beautify_hist_matplotlib(fig : plt.Figure, ax : plt.Axes, results : dict, histogram : Iterable, bins : Iterable, matplotlib_config : dict) -> tuple :
    """
    Add beautification to the matplotlib histogram plot.

    Parameters
    ----------
    fig : plt.Figure
        The matplotlib figure object.
    ax : plt.Axes
        The matplotlib axes object.
    results : dict
        The results dictionary containing the histogram data and statistics.
    histogram : list or np.ndarray
        The histogram values.
    bins : list or np.ndarray
        The bin edges.
    matplotlib_config : dict
        The matplotlib configuration dictionary.

    Returns
    -------
    fig : plt.Figure
        The matplotlib figure object.
    ax : plt.Axes
        The matplotlib axes object.
    """

    # (OPTIONAL) Normalize histogram and modify y-axis labels
    if matplotlib_config['normalize_hist'] :
        yticks_labels = [float(item.get_text()) for item in ax.get_yticklabels()]
        yticks_labels = [f"{item * 100:.1f}%" for item in yticks_labels]
        ax.set_yticklabels(yticks_labels)

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


    # Set x-axis scale
    if matplotlib_config['bins_distribution'] == 'logarithmic' : ax.set_xscale('log')

    # Add xticks
    ax.set_xticks(bins)
    ax.set_xticklabels(np.round(bins, 1))

    # Add x-axis label
    ax.set_xlabel(results['bins_variable'])

    # Set y-axis scale
    if matplotlib_config['y_axis_log'] : ax.set_yscale('log')
    
    # Add y-axis label
    if matplotlib_config['normalize_hist'] :
        ax.set_ylabel('Proportion of samples')
    else :
        ax.set_ylabel('Number of samples')

    # Add Grid
    if matplotlib_config['add_grid'] :
        ax.set_axisbelow(True)
        ax.grid(True)

    # Adjust limits
    ax.set_ylim([0, max(histogram) * 1.1])

    fig.tight_layout()

    return fig, ax


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# streamlit backend

def draw_hist_streamlit(streamlit_container) :

    results = load_data_for_plotting()

    plot_type = st.session_state.plot_type if 'plot_type' in st.session_state else "Type 1"

    if plot_type == "Type 1" :
        x_label = results['bins_variable']

        if min(results['histogram']) >= 0 and max(results['histogram']) <= 1 :
            y_label = 'Proportion of samples'
        else :
            y_label = 'Number of samples'

        results_df = pd.DataFrame({
            "labels" : results['labels'],
            "histogram" : results['histogram'],
        })

        with streamlit_container :

            st.bar_chart(
                data = results_df,
                x = 'labels', y = 'histogram',
                x_label = x_label, y_label = y_label,
                color = st.session_state.color,
            )
    elif plot_type == "Type 2" :
        pass
    elif plot_type == "Type 3" :
        results_UC = results['results_UC']
        results_CD = results['results_CD']
        results_control = results['results_control']

        x_label = results_UC['bins_variable']

        if min(results_UC['histogram']) >= 0 and max(results_UC['histogram']) <= 1 :
            y_label = 'Proportion of samples'
        else :
            y_label = 'Number of samples'

        results_df = pd.DataFrame(
            data = np.array([results_UC['histogram'], results_CD['histogram'], results_control['histogram']]).T,
            columns = ["UC", "CD", "control"]
        )

        results_df.insert(0, 'labels', results_UC['labels'])

        with streamlit_container :

            st.bar_chart(
                data = results_df,
                x = 'labels', y = ['UC', 'CD', 'control'],
                x_label = x_label, y_label = y_label,
            )

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Other functions

def get_color_hex() :
    color_hex = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    # List without blue, orange, and green (the first three colors of the default matplotlib color cycle)
    # Used to avoid confusion when plotting multiple histograms with Type 3 plot.
    # In type 3 I used the default color cycle of matplotlib, which starts with blue, orange, and green.
    color_hex = ['#17becf', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22']

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

def load_data_for_plotting() -> dict :
    # Get the name of variable to use for the plot
    bins_variable = st.session_state.bins_variable

    if ":" in bins_variable : bins_variable = bins_variable.split(":")[1].strip()

    # Get the type of plot
    plot_type = st.session_state.plot_type

    if plot_type == 'Type 1' :
        path_to_results = f'./results/{bins_variable}/results_all.pkl'

        results = load_data_from_pkl_file(path_to_results)
    elif plot_type == 'Type 2' or plot_type == 'Type 3' :
        class_list = ['UC', 'CD', 'control', 'all']

        results = dict()

        for class_to_load in class_list :
            path_to_results = f'./results/{bins_variable}/results_{class_to_load}.pkl'

            results[f'results_{class_to_load}'] = load_data_from_pkl_file(path_to_results)
    
    return results

def load_data_from_pkl_file(path_to_results : str = 'all') -> dict :
    # Load the results
    with open(path_to_results, 'rb') as f :
        results = pickle.load(f)
    
    # Compute the labels for the bins
    x_labels = []
    bins = results['bins']
    for i in range(len(bins) - 1) : x_labels.append(f"{round(bins[i], 1)}-{round(bins[i + 1], 1)}")
    results['labels'] = x_labels

    return results
