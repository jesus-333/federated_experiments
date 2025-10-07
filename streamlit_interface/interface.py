"""

@author: Alberto Zancanaro (Jesus)
@organization: Luxembourg Centre for Systems Biomedicine (LCSB)
@contact : alberto.zancanaro@uni.lu
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Imports

import streamlit as st
import pandas as pd

import support

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

st.set_page_config(
    page_title = "Clinnova Federated Hist PoC",
    layout="wide"
)

debug = True
column_gap = 'medium'
column_proportions = [0.5, 0.5]

# with st.sidebar :
#     hist_computation_config = support.build_hist_computation_options()

# hist_canvas = st.container()
hist_canvas = st.columns([1])[0]

hist_computation_option_column, hist_plot_column = st.columns(column_proportions, border = debug, gap = column_gap)

with hist_computation_option_column :
    hist_computation_config = support.build_hist_computation_options()

with hist_plot_column :

    genre = st.radio(
        label = "Plot type",
        options = ["Type 1", "Type 2", "Type 3"],
        captions = ["", "", ""],
        on_change = None,
    )

    bins_variable = st.selectbox(
        label = 'Select the plot backend',
        options = ['matplotlib', 'streamlit'], index = 0,
        key = 'plot_backend',
    )

    if bins_variable == 'matplotlib' :
        plot_config = support.build_hist_plot_options_matplotlib(hist_canvas)
    elif bins_variable == 'streamlit' :
        plot_config = support.build_hist_plot_options_streamlit(hist_canvas)
