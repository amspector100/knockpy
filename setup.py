import numpy as np
import os.path
import codecs
import setuptools
import distutils
from setuptools import find_packages, setup
from distutils.extension import Extension
from Cython.Build import cythonize

with open("README.md", "r") as fh:
    long_description = fh.read()

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

ext_modules = [
    Extension(
        "knockpy.mlr._update_hparams",
        sources=["knockpy/mlr/_update_hparams.pyx"],
    ),
    Extension(
        "knockpy.mlr._mlr_spikeslab_mx",
        sources=["knockpy/mlr/_mlr_spikeslab_mx.pyx"],
    ),
    Extension(
        "knockpy.mlr._mlr_spikeslab_fx",
        sources=["knockpy/mlr/_mlr_spikeslab_fx.pyx"],
    )
]

# ### Allows installation if cython is not yet installed
# try:
#     from Cython.Build import cythonize
# except ImportError:
#     # create closure for deferred import
#     def cythonize (*args, ** kwargs ):
#         from Cython.Build import cythonize
#         return cythonize(*args, ** kwargs)

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
        "numpy>=1.19",
        "cython>=0.29.21",
        "scipy>=1.5.2",
        "cvxpy>=1.0.25", 
        "scikit_learn>=0.22",
        "networkx>=2.4",
        "tqdm>=4.36.1",
    ],
    extras_require={
        "kpytorch":["torch>=1.4.0"],
        "fast":["choldate", "scikit-dsdp"]
    },
    setup_requires=[
        'numpy>=1.19',
        'setuptools>=58.0',
        'cython>=0.29.21',
    ],
    ext_modules=cythonize(
        ext_modules,
        compiler_directives={
            "language_level": 3, 
            "embedsignature": True
        },
        annotate=False,
    ),
    include_dirs=[np.get_include()],
)