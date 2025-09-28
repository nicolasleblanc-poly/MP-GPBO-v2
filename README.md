I modified the code for both the synthetic problems and the neurostimulation ones.
- The file Experiment_benchmarks_NL_Feb26.ipynb is used to run the synthetic problems.
- The file neurostim.ipynb is used to run on the neurostimulation data. This file uses BO_WITHExtensiveSearch.py, which is identical to BO.py, except that I have added extensive search functionality. I wanted to keep the BO files separate between the two problems.

You can download the non-human primate NHP data from this OSF link (It is given in Marco's Cell Report Medicine article): https://osf.io/54vhx/files/osfstorage

The benchmarks I added are:
- Py NOMAD, which is a Python package of the derivative-free optimization library NOMAD.
-> Download link: https://pypi.org/project/PyNomadBBO/
-> General NOMAD GitHub link: https://github.com/bbopt/nomad

- Extensive search (ES) for the neurostimulation problems. Marco had written it in Matlab, so I rewrote it in Python.
- Simulated annealing (SA). The performance is too good. We should revisit the number of steps used for optimizing the next query by adjusting the temperature parameter.

- SMAC, a hyperparameter tuning library that is compatible with discrete domains: https://github.com/automl/SMAC3

- Optuna, a hyperparameter tuning library that is compatible with discrete domains: https://github.com/optuna/optuna

My Python version is 3.11.9
