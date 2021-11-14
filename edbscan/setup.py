"""Compile the Cython _inner code."""

from distutils.core import Extension, setup

import numpy as np
from Cython.Build import cythonize

if __name__ == "__main__":
    ext = Extension(
        "edbscan._inner",
        sources=["edbscan/_inner.pyx"],
        language="c++",
    )
    setup(
        name="_inner",
        ext_modules=cythonize(ext, annotate=True),
        include_dirs=[np.get_include()],
    )
