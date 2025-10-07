#!/bin/sh
# 
# Author : Alberto  Zancanaro (Jesus)
# Organization: Luxembourg Centre for Systems Biomedicine (LCSB)
# Contact : alberto.zancanaro@uni.lu

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Settings

path_server_config="./config/server_config.toml"
path_to_save_results="./results/"

path_plot_config="./config/plot_config.toml"

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

# Run flower app on the remote federation
flwr run --stream --run-config "path_server_config=\"${path_server_config}\"" ./ remote-hist 

# Download the results
# (Temporary solution. In future the flwr pull command )
rsync -avzh -e "ssh -p 8022" clinnova_vm_server:./test_server_app/results/ ${path_to_save_results}

# Plot the results
python ./scripts/plot_results.py --path_results_file "${path_to_save_results}results.pkl" --path_plot_config ${path_plot_config}
