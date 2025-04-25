# Knockpy

A python implementation of the knockoffs framework for variable selection. See <https://amspector100.github.io/knockpy/> for detailed documentation and tutorials.

## Installation

To install and begin using knockpy, simply enter:

```bash
pip install knockpy[fast]
```

To use the (optional) kpytorch submodule, you will need to install [pytorch](https://pytorch.org/).

### What if installation fails?

knockpy relies on heavy-duty linear algebra routines which sometimes fail on non-Linux environments.

1. To start, install a lightweight version of knockpy using
`pip install knockpy`. This should install correctly on all devices, and contains nearly all of the functionality of the prior installation. However, the algorithms for computing optimal distributions for Gaussian knockoffs, such as [minimum reconstructability knockoffs](https://arxiv.org/abs/2011.14625) and [SDP knockoffs](https://arxiv.org/abs/1610.02351), may be an order of magnitude slower.

2. [Optional] To speed up computation for minimum reconstructability knockoffs (the default knockoff type):

    (a) Run

    ```bash
    pip install cython>=0.29.14`
    ```

    If the installation fails, likely due to the incorrect configuration of a C compiler, you have three options. First, the [Anaconda](https://docs.anaconda.com/anaconda/user-guide/tasks/install-packages/) package manager includes a compiler, so the command

    ```bash
    install cython
    ```

    should work on all platforms. Second, on Windows, you can install precompiled binaries for cython [here](https://www.lfd.uci.edu/~gohlke/pythonlibs/). Lastly, on all platforms, the documentation [here](https://cython.readthedocs.io/en/latest/src/quickstart/install.html) describes how to properly configure a C compiler during installation.

    (b) Run

    ```bash
    pip install git+https://github.com/jcrudy/choldate.git
    ```

3. [Optional] To speed up computation for (non-default) SDP knockoffs, you will need to install `scikit-dsdp`. This can be challenging on non-Linux environments. We hope to provide more explicit instructions for installation of this package in the future.

## Quickstart

Given a data-matrix `X` and a response vector `y`, knockpy makes it easy to use knockoffs to perform variable selection using a wide variety of machine learning algorithms (also known as "feature statistic") and types of knockoffs. One quick example is shown below, where we use the cross-validated lasso to assign variable importances to the features and knockoffs.

```python
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

## Development

To install knockpy for development, you must first install [uv](https://github.com/astral-sh/uv).

```bash
git clone https://github.com/amspector100/knockpy.git
cd knockpy
make install-pre-commit
```

The `Makefile` makes it easy to perform the most common operations:

* `make check-all` runs linting and `uv.lock` checks
* `make check-lint` checks for linting issues
* `make check-lock` verifies the `uv.lock` is aligned to `pyproject.toml`
* `make clean` cleans the virtual environment and caches
* `make default` runs a default set of checks on the code
* `make fix-all` formats the code, fixes lint errors and runs locks `uv.lock` to `pyproject.toml`
* `make fix-format` formats the code
* `make fix-lint` fixes linting issues
* `make fix-lint-unsafe` fixes linting issues potentially adding inadvertant bugs
* `make help` outputs the different make options
* `make install` build install the distribution
* `make install-pre-commit` installs pre-commit hooks
* `make lock` locks `uv.lock` to `pyproject.toml`
* `make install-pre-commit` installs pre-commit hooks
* `make run-tests` runs the unit tests
* `make sync` syncs the python environment with `uv.lock`

`.vscode/settings.json` is set so that unit tests can be run without further configuration.

### Releasing to pypi

If you are an owner of the knockpy repo, you can publish a new version to pypi by:

1. Selecting the [Bump Version and Start New Release workflow](https://github.com/amspector100/knockpy/actions/workflows/pages/bump-version.yml)
2. Selecting `Run workflow`
3. Ensure that the `main` branch is selected
4. Pressing `Run workflow`
5. This should trigger a new build and release in the [Make a New Release](https://github.com/amspector100/knockpy/actions/workflows/bump-version.yml) workflow page. This will publish to pypi.org.
6. Select the new release in the [Releases](https://github.com/amspector100/knockpy/releases) page
7. Add release notes and publish the release.

## Reference

If you use knockpy in an academic publication, please consider citing [Spector and Janson (2020)](https://arxiv.org/abs/2011.14625). The bibtex entry is below:

```latex
@article{AS-LJ:2020,
  title={Powerful Knockoffs via Minimizing Reconstructability},
  author={Spector, Asher and Janson, Lucas},
  journal={Annals of Statistics},
  year={2021+},
  note={To Appear}
}
```