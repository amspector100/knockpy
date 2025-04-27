import os

import numpy as np
import setuptools
from setuptools import Extension

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
    ),
]

CYTHONIZE = cythonize is not None

if CYTHONIZE:
    compiler_directives = {"language_level": 3, "embedsignature": True}
    ext_modules = cythonize(
        ext_modules, compiler_directives=compiler_directives, annotate=False
    )
else:
    ext_modules = no_cythonize(ext_modules)

setuptools.setup(
    ext_modules=ext_modules,
    include_dirs=[np.get_include()],
)
