#!/bin/sh
# 
# Author : Alberto  Zancanaro (Jesus)
# Organization: Luxembourg Centre for Systems Biomedicine (LCSB)
# Contact : alberto.zancanaro@uni.lu

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Settings

path_server_config="./config/server_config.toml"

path_plot_config="./config/plot_config.toml"

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

flwr run --stream --run-config "path_server_config=\"${path_server_config}\"" ./ local-deployment 

python ./scripts/plot_results.py --path_results_file "./results/results.pkl" --path_plot_config ${path_plot_config}
