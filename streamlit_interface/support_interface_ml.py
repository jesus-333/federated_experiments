"""
Support function used for the streamlit interface (ml algorithm)

@author: Alberto Zancanaro (Jesus)
@organization: Luxembourg Centre for Systems Biomedicine (LCSB)
@contact : alberto.zancanaro@uni.lu
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Imports

import streamlit as st
import subprocess

import support_plot_ml

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def build_ml_computation_options(streamlit_container_for_the_plot) :
    st.subheader("Training Settings", divider = True)
    st.selectbox(
        label = 'Select the ML model',
        options = ['SVM'], index = 0,
        key = 'ml_model_name',
        args = [streamlit_container_for_the_plot]
    )

    st.button(
        label    = "Compute Histogram",
        key      = "compute_hist_button",
        on_click = train_ml_model,
        args = [streamlit_container_for_the_plot]
    )

def build_ml_plot_options(streamlit_container_for_the_plot) :
    st.subheader("Plot Settings", divider = True)

    st.write("Class color")
    color_column_list = st.columns([0.33, 0.33, 0.33])
    
    for i in range(3) :
        with color_column_list[i] :
            st.selectbox(
                label = f"Color class {i}",
                options = support_plot_ml.get_color_hex(), index = i,
                format_func = support_plot_ml.get_color_name_from_hex,
                key = f'color_class_{i}',
                on_change = support_plot_ml.plot_decision_boundary,
                args = [streamlit_container_for_the_plot]
            )

    st.radio(
        label = "Training mode",
        options = ["FL training", "Only client 1", "Only client 2"],
        captions = [
            "Show the decision boundary obtained with the FL training",
            "Show the decision boundary if only data from client 1 were used for the training",
            "Show the decision boundary if only data from client 2 were used for the training",
        ],
        key = 'type_of_params',
        on_change = support_plot_ml.plot_decision_boundary,
        args = [streamlit_container_for_the_plot]
    )

    st.slider(
        label = "Grid Resolution",
        min_value = 0.01, max_value = 1., step = 0.01,
        value = 0.05,
        key = 'grid_resolution',
    )

    st.slider(
        label = "Grid Padding (%)",
        min_value = 0., max_value = 1., step = 0.01,
        value = 0.05,
        key = 'grid_padding',
    )

    st.selectbox(
        label = 'Dimensionality Reduction Method',
        options = ['PCA', 't-SNE'], index = 0,
        key = 'dimensionality_reduction',
        on_change = support_plot_ml.plot_decision_boundary,
        args = [streamlit_container_for_the_plot]
    )


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def train_ml_model(streamlit_container_for_the_plot) :

    subprocess.call(['sh', './other_scripts/run_ml_app.sh'])

