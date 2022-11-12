import numpy as np
import os
import codecs
import setuptools
from setuptools import Extension
#from distutils.extension import Extension

try:
    from Cython.Build import cythonize
except ImportError:
    cythonize = None

# https://cython.readthedocs.io/en/latest/src/userguide/source_files_and_compilation.html#distributing-cython-modules
def no_cythonize(extensions, **_ignore):
    for extension in extensions:
        sources = []
        for sfile in extension.sources:
            path, ext = os.path.splitext(sfile)
            if ext in (".pyx", ".py"):
                if extension.language == "c++":
                    ext = ".cpp"
                else:
                    ext = ".c"
                sfile = path + ext
            sources.append(sfile)
        extension.sources[:] = sources
    return extensions

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
        "knockpy.mlr._mlr_spikeslab",
        sources=["knockpy/mlr/_mlr_spikeslab.pyx"],
    ),
    Extension(
        "knockpy.mlr._mlr_spikeslab_group",
        sources=["knockpy/mlr/_mlr_spikeslab_group.pyx"],
    ),
    Extension(
        "knockpy.mlr._mlr_spikeslab_fx",
        sources=["knockpy/mlr/_mlr_spikeslab_fx.pyx"],
    ),
    Extension(
        "knockpy.mlr._mlr_oracle",
        sources=["knockpy/mlr/_mlr_oracle.pyx"],
    )
]

#CYTHONIZE = bool(int(os.getenv("CYTHONIZE", 0))) and cythonize is not None
CYTHONIZE = cythonize is not None

if CYTHONIZE:
    compiler_directives = {"language_level": 3, "embedsignature": True}
    ext_modules = cythonize(
        ext_modules, compiler_directives=compiler_directives, annotate=False
    )
else:
    ext_modules = no_cythonize(ext_modules)

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
        "fast":["choldate", "scikit-dsdp"],
        "docs": ["sphinx", "sphinx-rtd-theme"],
    },
    setup_requires=[
        'numpy>=1.19',
        'setuptools>=58.0',
        #'cython>=0.29.21',
    ],
    ext_modules=ext_modules,
    include_dirs=[np.get_include()],
)