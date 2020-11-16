import setuptools
import codecs
import os.path

def read(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    with codecs.open(os.path.join(here, rel_path), 'r') as fp:
        return fp.read()

def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith('__version__'):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    else:
        raise RuntimeError("Unable to find version string.")

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="knockpy",
    version=get_version('knockpy/__init__.py'),
    author="Asher Spector",
    author_email="amspector100@gmail.com",
    description="Knockoffs for variable selection",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/amspector100/knockpy",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[
        "choldate",
        "cython>=0.29.14",
        "numpy>=1.17.4",
        "scipy>=1.5.2",
        "cvxpy>=1.0.25",
        "matplotlib>=3.1.0",
        "scikit_learn>=0.22",
        "statsmodels>=0.10.1",
        "scikit-dsdp",
        "networkx>=1.11",
        "tqdm>=4.36.1",
        "group_lasso",
        "pyglmnet"
    ],
    extras_require={
        "kpytorch":["torch>=1.4.0"],
    }
)