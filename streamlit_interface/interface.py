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

# hist_canvas = st.container(key = 'hist_canvas')
hist_canvas = st.columns([1], border = debug)[0]

hist_computation_option_column, hist_plot_column = st.columns(column_proportions, border = debug, gap = column_gap)


with hist_plot_column :

    genre = st.radio(
        label = "Plot type",
        options = ["Type 1", "Type 2", "Type 3"],
        captions = ["", "", ""],
        on_change = None,
    )

    st.write("Other options")
    normalize_hist = st.checkbox('Normalize hist', key = 'normalize_hist', value = False)

    plot_backend = st.selectbox(
        label = 'Select the plot backend',
        options = ['matplotlib', 'streamlit'], index = 0,
        key = 'plot_backend',
        on_change = support.draw_hist,
        args = [hist_canvas] 
    )

    if plot_backend == 'matplotlib' :
        plot_config = support.build_hist_plot_options_matplotlib(hist_canvas)
    elif plot_backend == 'streamlit' :
        plot_config = support.build_hist_plot_options_streamlit(hist_canvas)

with hist_computation_option_column :
    hist_computation_config = support.build_hist_computation_options(hist_canvas)
