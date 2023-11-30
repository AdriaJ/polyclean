Time benchmark experiment
=========================

Code to run the experiment of the section 6.1 in the article.

1. How to reproduce the results?
--------------------------------

To launch the computations, you need to run the following commands::

    > cd scripts/benchmark
    > ./run_exp.sh

This command produces two files `metrics.csv` and `props.csv` in the current folder. The first one stores the
reconstruction times while the second one keeps various information about the experiment (dimension, seed,...).
The parameters of the experiment are provided on the first 3 lines of the file `run_exp.sh`, and need to be changed
manually in the file::

    rmax=(300 600 900) or e.g. 1000
    nreps=1
    save="true"

If the argument `save` is set tu `"true"`, the reconstructed images are saved in the folder `reconstructions/`.

The benchmark plots provided in the PolyCLEAN article can be reproduced with the following command, after having
entered the `df_dir_path` and `exp_name` of the data to plot::

    > python open_df.py


2. Description of the files
----------------------------

- `run_exp.sh`
    Bash script to run the full experiment experiment.
- `fill_one_row.sh`
    Bash script to run one single reconstruction. This script runs the 4 following files, in addition to
    `WS-CLEAN`_. The dataframes in `metrics.csv` and `props.csv` are filled by this script.
- `simulate_pb.py`
    Python script to simulate a problem (source and measurements).
- `solve_lasso.py`
    Python script to solve the LASSO problem with PolyCLEAN and APGD. The reconstruction are stored in a
    temporary folder.
- `pp_wsclean.py`
    Python script to postprocess the output of ws-clean.
- `fill_df.py`
    Python script to open and fill the dataframes `metrics.csv` and `props.csv`.
- `open_df.py`
    Additional Python script to load the dataframes and plot the results.
- `config.yaml`
    Configuration file for the simulations and all the reconsruction methods.


.. _WS-CLEAN: https://wsclean.readthedocs.io/en/latest/