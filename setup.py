"""Setup module for this Python package."""

import os
import re

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

setup(
    name="edbscan",
    version="0.0.0",
    description="Enforced Density -Based Spatial Clustering of Applications with Noise.",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=install_requires,
    include_package_data=True,
)
