"""
@author: Alberto Zancanaro (Jesus)
@organization: Luxembourg Centre for Systems Biomedicine (LCSB)
@contact : alberto.zancanaro@uni.lu
@date: September 2025
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

import argparse
import numpy as np
import os
import pandas as pd

import support_scripts

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# Create parser
parser = argparse.ArgumentParser(description = 'Prepare the data for the FL hist computation')

parser.add_argument('--n_clients', type = int, default = 2        , help = 'Number of clients to split the data. Default is 2.')
parser.add_argument('--path_data', type = str, default = None     , help = 'Path to the original data. If not provided an error is raised. Deafault is None.')
parser.add_argument('--path_save', type = str, default = './data/', help = 'Path to save the splitted data. Default is ./data/.')
parser.add_argument('--seed'     , type = int, default = 42       , help = 'Random seed to use for the data splitting. It must be a positive integer. If an invalid value is provided the default value will be used. Default is 42.')
parser.add_argument('--keep_labels_proportion', action = 'store_true', default = False, help = 'If passed as argument the labels proprtion is kept in each client. Default is False.')

args = parser.parse_args()

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Settings

if args.path_data is None :
    raise ValueError('Path to the original data not provided. Please provide it using the argument --path_data.')
else :
    path_data = args.path_data

if args.n_clients < 2 :
    raise ValueError(f'Number of clients must be at least 2. Provided value is {args.n_clients}.')
else :
    n_clients = args.n_clients

seed = args.seed if args.seed >= 0 else 42

keep_labels_proportion = args.keep_labels_proportion

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Data loading

# Load data
dataset_pd = pd.read_excel(path_data)

# Get the name of the features (first column)
features_name = dataset_pd['# Feature / Sample']

# Get data (all other columns)
data_pd = dataset_pd.iloc[:, 1:]

# Get labels
labels = data_pd.iloc[2]

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# Create the folder to save the splitted data if it does not exist
os.makedirs(args.path_save, exist_ok = True)

for client_id in range(n_clients):
    # Split the data for each client
    data_per_client, labels_per_client = support_scripts.split_data_for_clients_uniformly(data_pd.to_numpy().T, n_clients, args.seed, labels.to_numpy(), keep_labels_proportion)

    # Create a DataFrame for each client
    data_per_client_df = pd.DataFrame(data_per_client[client_id], columns = features_name)

    # Save the data for each client in csv format
    path_save_data_client = os.path.join(args.path_save, f'client_{client_id}_data.csv')
    data_per_client_df.to_csv(path_save_data_client, index = False)

