"""

@author: Alberto Zancanaro (Jesus)
@organization: Luxembourg Centre for Systems Biomedicine (LCSB)
@contact : alberto.zancanaro@uni.lu
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Imports

import streamlit as st

import support_interface

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

st.set_page_config(
    page_title = "Clinnova Federated Hist PoC",
    layout="wide"
)

debug = True
column_gap = 'medium'

# with st.sidebar :
#     hist_computation_config = support.build_hist_computation_options()

# hist_canvas = st.container(key = 'hist_canvas', width = 'stretch', height = 400)
hist_canvas = st.columns([1], border = debug)[0]

column_proportions = [0.5, 0.5]
hist_computation_option_column, hist_plot_column = st.columns(column_proportions, border = debug, gap = column_gap)

with hist_computation_option_column :
    hist_computation_config = support_interface.build_hist_computation_options(hist_canvas)

with hist_plot_column :
    support_interface.build_hist_plot_options(hist_canvas)

