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
import support_interface_hist

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
        options = ['None', 'PCA', 't-SNE'], index = 0,
        key = 'dimensionality_reduction',
        on_change = support_plot_ml.plot_decision_boundary,
        args = [streamlit_container_for_the_plot]
    )

    build_options_dimensionality_reduction(st.session_state.dimensionality_reduction, streamlit_container_for_the_plot)

def build_options_dimensionality_reduction(dimensionality_reduction_methods : str, streamlit_container_for_the_plot) :
    if dimensionality_reduction_methods == 'None' :
        clf_variables = support_interface_hist.read_txt_list("./streamlit_interface/field_hist.txt")

        # Create column for the two variable to plot
        column_variable_1, column_variable_2 = st.columns([0.5, 0.5])

        with column_variable_1 :
            st.selectbox(
                label = 'Variable 1',
                options = clf_variables, index = 0,
                key = 'clf_variable_1',
                on_change = support_plot_ml.plot_decision_boundary,
                args = [streamlit_container_for_the_plot]
            )

        with column_variable_2 :
            st.selectbox(
                label = 'Variable 2',
                options = clf_variables, index = 1,
                key = 'clf_variable_2',
                on_change = support_plot_ml.plot_decision_boundary,
                args = [streamlit_container_for_the_plot]
            )
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def train_ml_model(streamlit_container_for_the_plot) :

    subprocess.call(['sh', './other_scripts/run_ml_app.sh'])

