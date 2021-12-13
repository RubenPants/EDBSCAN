"""Setup module for this Python package."""

import os
import re
from distutils.core import Extension

import numpy as np
from Cython.Build import cythonize
from setuptools import find_packages, setup

# Fill `install_requires` with packages in environment.run.yml.
install_requires = []
with open(os.path.join(os.path.dirname(__file__), "environment.run.yml")) as spec:
    for line in spec:
        match = re.search(r"^\s*-\s+(?P<n>.+)(?P<v>(?:~=|==|!=|<=|>=|<|>|===|@)[^\s\n\r]+)", line)
        if match and match.group("n") not in ("pip", "python"):
            # Support git+ssh://git@.../pkg.git@vx.y.z packages, see stackoverflow.com/a/54794506.
            prefix = (
                match.group("n").split("/")[-1].replace(".git", "") + " @ "
                if match.group("n").startswith("git+")
                else ""
            )
            install_requires.append(prefix + match.group("n") + match.group("v"))

with open("README.md", "r") as f:
    long_description = f.read()

if __name__ == "__main__":
    # Specify the Cython extension
    ext = Extension(
        "edbscan._inner",
        sources=["src/edbscan/_inner.pyx"],
        language="c++",
    )

    # Run the setup
    setup(
        name="edbscan",
        version="0.0.0",
        package_dir={"": "src"},
        packages=find_packages(where="src"),
        author="Ruben Broekx",
        description="Enforced Density -Based Spatial Clustering of Applications with Noise.",
        long_description=long_description,
        long_description_content_type="text/markdown",
        url="https://github.com/RubenPants/EDBSCAN",
        ext_modules=cythonize(ext, annotate=True),
        include_dirs=[np.get_include()],
        install_requires=install_requires,
        # include_package_data=True,
    )
