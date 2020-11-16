# Knockpy

A python implementation of the knockoffs framework for variable selection.

## Installation

To install knockpy, first install choldate:
``pip install git+git://github.com/jcrudy/choldate.git``
Then, install knockpy using pip:
``pip install knockpy``
To use the (optional) kpytorch submodule, you will need to install [pytorch](https://pytorch.org/). If the installation fails on your system, please reach out to me and I'll try to help.

## Basic Usage



## Modularity

## To run tests

- To run all tests, run ``python3 -m pytest`` 
- To run a specific label, run ``pytest -v -m {label}``.
- To select all labels except a particular one, run ``pytest -v -m "not {label}"`` (with the quotes).
- To run a specific file, try pytest test/{file_name}.py. To run a specific test within the file, run pytest test/{file_name}.py::classname::test_method. You also don't have to specify
the exact test_method, you get the idea.
- To run a test with profiling, try ``python3 -m pytest {path} --profile``. This should generate a set of .prof files in prof/. Then you can run snakeviz filename.prof to visualize the output.
There are also more flags/options for outputs in the command line command.
- Alternatively, cprofilev is much better.
To run cprofilev, copy and paste the test to proftest/* and then run 
``python3 -m cprofilev proftest/test_name.py``.
