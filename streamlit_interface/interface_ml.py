"""

@author: Alberto Zancanaro (Jesus)
@organization: Luxembourg Centre for Systems Biomedicine (LCSB)
@contact : alberto.zancanaro@uni.lu
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Imports

import streamlit as st

import support_interface_ml

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

st.set_page_config(
    page_title = "Clinnova Federated Hist PoC",
    layout="wide"
)

debug = True
column_gap = 'medium'

st.header("Clinnova User Portal", divider = True)

# with st.sidebar :
#     hist_computation_config = support.build_hist_computation_options()

# hist_canvas = st.container(key = 'hist_canvas', width = 'stretch', height = 400)
hist_canvas = st.columns([1], border = debug)[0]

column_proportions = [0.5, 0.5]
ml_training_options_column, ml_plot_options_column = st.columns(column_proportions, border = debug, gap = column_gap)

with ml_training_options_column :
    hist_computation_config = support_interface_ml.build_ml_computation_options(hist_canvas)

with ml_plot_options_column :
    support_interface_ml.build_ml_plot_options(hist_canvas)

st.write("---")
