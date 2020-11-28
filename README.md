# Knockpy

A python implementation of the knockoffs framework for variable selection. See https://amspector100.github.io/knockpy/ for detailed documentation and tutorials.

## Installation

To install knockpy, first install choldate using the following command:

``pip install git+git://github.com/jcrudy/choldate.git``

Note that installing choldate does require installing cython, which requires access to a C compiler. This can be challenging on Windows, and if you have difficulty on any OS, please carefully follow the instructions [here](https://cython.readthedocs.io/en/latest/src/quickstart/install.html).

Then, install knockpy using pip:

``pip install knockpy``

If the installation fails on your system, please reach out to me and I'll try to help.

To use the (optional) kpytorch submodule, you will need to install [pytorch](https://pytorch.org/).

## Quickstart

Given a data-matrix `X` and a response vector `y`, knockpy makes it easy to use knockoffs to perform variable selection using a wide variety of machine learning algorithms (also known as "feature statistic") and types of knockoffs. One quick example is shown below, where we use the cross-validated lasso to assign variable importances to the features and knockoffs.  

```
    import knockpy as kpy
    from knockpy.knockoff_filter import KnockoffFilter

    # Generate synthetic data from a Gaussian linear model
    data_gen_process = kpy.dgp.DGP()
    data_gen_process.sample_data(
        n=1500, # Number of datapoints
        p=500, # Dimensionality
        sparsity=0.1,
        x_dist='gaussian',
    )
    X = data_gen_process.X
    y = data_gen_process.y
    Sigma=data_gen_process.Sigma

    # Run model-X knockoffs
    kfilter = KnockoffFilter(
        fstat='lasso',
        ksampler='gaussian',
    )
    rejections = kfilter.forward(X=X, y=y, Sigma=Sigma)
```

Most importantly, ``knockpy`` is built to be modular, so researchers and analysts can easily layer functionality on top of it.

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
