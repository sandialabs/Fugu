import os

from setuptools import find_packages, setup

package_list = find_packages()

# Get the list of dependencies
base_dependencies = [
    "networkx==3.2.1",
    "numpy",
    "pandas~=2.2.3",
    "pyyaml",
    "pytest~=8.3.4",
    "scipy",
]

# Check for an environment variable to include additional dependencies
additional_dependencies = os.getenv("INCLUDE_DEPENDENCIES", "").split(",")

# Check for an environment variable to exclude specific dependencies
excluded_dependencies = os.getenv("EXCLUDE_DEPENDENCIES", "").split(",")

# Remove excluded dependencies from the base list
filtered_dependencies = [dep for dep in base_dependencies if not any(omit_dep in dep for omit_dep in excluded_dependencies if omit_dep != "")]

# Add additional dependencies to filtered dependencies
final_dependencies = filtered_dependencies + [add_dep for add_dep in additional_dependencies if add_dep != ""]

setup(
    name="fugu",
    version="1.4.2",
    description="A python library for computational neural graphs",
    install_requires=final_dependencies,
    extras_require={
        "whetstone": ["tensorflow==2.18.0", "keras==3.8.0"],
        "dev": ["pre-commit", "isort", "black", "tqdm", "tox", "tox-conda", "coverage"],
        "examples": ["notebook", "matplotlib", "tqdm"],
    },
    packages=package_list,
    python_requires=">=3.9, <3.12",
)
