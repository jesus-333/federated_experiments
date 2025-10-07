Proof of concept for the Clinnova Project of histrogram computation in a federated environment.

[Link](https://static-content.springer.com/esm/art%3A10.1038%2Fs41564-018-0306-4/MediaObjects/41564_2018_306_MOESM8_ESM.xlsx) of the dataset.

The work is inspired by a similar [flower tutorial](https://flower.ai/docs/examples/quickstart-pandas.html) (Note that the link was copied the 29/09/25).

# App summary

This section contains a brief description of the folders and their files
## `hist_app`

The `hist_app` folder contains the python implementation of the flower app.

- [`client_app.py`](./hist_app/client_app.py). Implementation of the client. Provide the data required by the server for the computation.
- [`server_app.py`](./hist_app/server_app.py). Implementation of the server. Request the min and max for each client in the first round and use these values to compute unified bins for the federation. In the second round sends bins to clients, receives histograms back, and calculates the overall histogram.


## `Script`

The `script` folder contains the scripts used to run the flower app for federated histogram computation.

### Python Scripts

### Shell scripts
