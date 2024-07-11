from setuptools import find_packages, setup

package_list = find_packages()

setup(
    name="fugu",
    version="0.1",
    description="A python library for computational neural graphs",
    install_requires=[
        "decorator~=4.4.2",
        "future~=0.18.2",
        "greenlet",
        "msgpack~=1.0.0",
        "networkx==2.4",
        "numpy<1.24.0",
        "pandas~=1.5.3",
        "python-dateutil~=2.8.1",
        "pytz~=2020.1",
        "six~=1.15.0",
        "furo~=2021.11.16",
        "pyyaml",
    ],
    extras_require={
        "whetstone": ["tensorflow<=2.10", "keras<=2.10", "scipy"],
    },
    packages=package_list,
)
