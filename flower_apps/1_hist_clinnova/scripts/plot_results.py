"""
Plot the histogram obtained from the flower app

@author: Alberto Zancanaro (Jesus)
@organization: Luxembourg Centre for Systems Biomedicine (LCSB)
@contact : alberto.zancanaro@uni.lu
@date: September 2025
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

import argparse
import matplotlib.pyplot as plt
import os
import pickle
import toml

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# Create parser
parser = argparse.ArgumentParser(description = 'Plot the histogram obtained from the flower app')

parser.add_argument('--path_results_file', type = str           , default = None , help = 'Path to the results file. If not provided an error is raised. Default is None.')
parser.add_argument('--path_plot_config' , type = str           , default = None , help = 'Path to a plot configuration file (in a toml format). If not provided default settings are used. Default is None.')
parser.add_argument('--path_save'        , type = str           , default = None , help = 'Path to save the plot. If not provided the plot will be shown but not saved. Default is None.')

args = parser.parse_args()

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# Settings

# Path to the results file
if args.path_results_file is None :
    raise ValueError('Path to the results file not provided. Please provide it using the argument --path_results_file.')
else :
    path_results_file = args.path_results_file

# Path to the plot configuration file
if args.path_plot_config is not None :
    # Check if the provided file is a toml file
    if not args.path_plot_config.endswith('.toml') :
        raise ValueError('Plot configuration file must be a toml file. Please provide a valid file using the argument --path_plot_config.')
    else :
        # Load plot configuration
        plot_config = toml.load(args.path_plot_config)
else :
    plot_config = dict(
        figsize = (10, 6),
        add_title = True,
        edgecolor = 'black',
        alpha = 0.7,
        add_grid = True,
        show_plot = True,

    )

# Path to save the plot
path_save = args.path_save if args.path_save is not None else None

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Load data

with open(path_results_file, 'rb') as f :
    results = pickle.load(f)

# Histogram data
bins = results['bins']
hist = results['histogram']
variable_name = results['bins_variable']

# Samples statistics
samples_mean = results['mean']
samples_std  = results['std']
n_samples    = results['n_samples']

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# Create the plot
fig, ax = plt.subplots(figsize = plot_config['figsize'])

# Plot histogram
ax.bar(bins[:-1], hist, width = (bins[1] - bins[0]), align = 'edge', edgecolor = plot_config['edgecolor'], alpha = plot_config['alpha'])

# Add plot info
ax.set_xlabel(variable_name)

if min(hist) >= 0 and max(hist) <= 1 :
    ax.set_ylabel('Proportion of samples')
else :
    ax.set_ylabel('Number of samples')

if plot_config['add_title'] : ax.set_title(f'Histogram of {variable_name}\nMean: {samples_mean:.2f}, Std: {samples_std:.2f}, N: {n_samples}')

if plot_config['add_grid'] : ax.grid(True)

fig.tight_layout()

if plot_config['show_plot'] :
    plt.show()

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# (OTIONAL) Save the plot

if 'path_save' in plot_config :
    os.makedirs(os.path.dirname(plot_config['path_save']), exist_ok = True)
    
    # Save the plot in the specified formats
    for extension in plot_config['image_file_format_list'] :
        # Create the full path
        full_path = f"{plot_config['path_save']}{plot_config['image_file_name']}{extension}"

        # Save the figure
        fig.savefig(full_path, format = extension)

        # Print info
        print(f'Plot saved to {full_path}')


