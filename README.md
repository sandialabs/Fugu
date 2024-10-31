# Fugu
A python library for computational neural graphs.

# Install

## Dependencies
A full list of dependencies is listed in requirements.txt.  The high level dependencies are:

- Numpy
- NetworkX
- Pandas

### A Note on Running Examples

Some of the examples additionally require Jupyter and matplotlib.


## Using Pip
```bash
git clone https://github.com/sandialabs/Fugu.git
cd Fugu
pip install --upgrade pip
pip install -e .[examples]
```

## Using Conda
Let's install the requirements for Fugu.
```bash
git clone https://github.com/sandialabs/Fugu.git
cd Fugu
conda env create -f fugu_conda_environment.yml
conda activate fugu
conda develop $PWD
```

## [OPTIONAL] STACS Backend
The Simulation Tool for Asynchronous Cortical Streams (STACS) is an optional simulator backend that may be used with Fugu. Developed to be parallel from the ground up, STACS leverages the highly portable Charm++ parallel programming framework. In addition to the parallel runtime, STACS also implements a memory-efficient distributed network data structure for network construction, simulation, and serialization. This provides a scalable simulation backend to support both large-scale and long running SNN experiments (e.g. on HPC systems).

While STACS is developed as a stand-alone simulator, through the use of a template network model and specially developed neuron and synapse models, users of Fugu may interface with STACS simply through Fugu's backend API. To utilize the STACS backend, some additional software tools are necessary. This section describes the installation of these tools from source for the STACS backend usage.

### Prerequisite Packages
First, install essential (linux) packages.
```
sudo apt-get install build-essential gfortran cmake cmake-curses-gui
sudo apt-get install libyaml-cpp-dev libfftw3-dev zlib1g-dev
sudo apt-get install mpich
```

### Charm++
Download Charm++ (v7.0.0) and untar the package.
```
wget http://charm.cs.illinois.edu/distrib/charm-7.0.0.tar.gz
tar -zxvf charm-7.0.0.tar.gz

mkdir charm
mv charm-7.0.0.tar.gz charm/
mv charm-v7.0.0/ charm/7.0.0
```

Build Charm++
```
cd charm/7.0.0
./build charm++ mpi-linux-x86_64 --with-production -j2
```

Add Charm++ to PATH variable (target=mpi-linux-x86_64 here)
```
export CHARM_ROOTDIR=/path/to/charm/version/target
export PATH=${CHARM_ROOTDIR}/bin:$PATH
```
Replace `/path/to/charm/version/target` with the directory path to your Charm++. Alternatively, add the above two lines to your `~/.bashrc` file for a variables to be persistent between linux sessions.

Next, install STACS repository
```
git clone https://github.com/sandialabs/STACS.git
cd STACS
make -j2
```

# Documentation
Documentation is currently spread across several files and directories.  We are working on including docstrings on all the classes and methods.

For now, you can check:
- https://sandialabs.github.io/Fugu/
- This `README.md`
- The `examples` folder


Additional documentation can be generated from the `docs` folder. Use the following instructions to generate it on your system.

```bash
pip install -U Sphinx
```
or
```bash
conda install Sphinx
```

Navigate to the `docs` folder.  Use the `sphinx-build` command with the `html` option to generate the HTML files on your system.
```
sphinx-build -b html -a source/ build/html
```
The documentation will be in `docs/build/html`.
Open `introduction.html`.  From this page you can you can navigate through the full website, Fugu Module, etc.

# Developers
The information below is for Fugu developers.

## Pre-commit hooks
Pre-commit is used to enforce a code standard thoughout Fugu's codebase. For now, the pre-commit hooks enforce 

1. Code formatting (via `black`)
2. Import sorting (via `isort`)

The following packages are required for the pre-commit hooks

1. black
2. isort
3. pre-commit

### Installation and setup

First install the necessary python packages using

`pip install black isort pre-commit`

or 

`conda install -c conda-forge black isort pre-commit`

Next, we install the pre-commit hooks with
`pre-commit install --install-hooks`
*Note:* `.pre-commit-config.yaml` must be present in the top level of Fugu for the pre-commit installation. For the pre-commit installs to work properly for me, I had to disconnect from the Sandia VPN. Lastly, these instructions have been tested using a conda environment.

Now the pre-commit hooks will be installed. The next time you commit a file to the repository, `isort` and `black` will check that the file conforms to the new code standards. If the file passes the checks then you will be prompted to enter a commit message. Otherwise, the pre-commit will display if one or both commands failed. To fix the problem run one or both of these commands on the culprit file:

`isort /path/to/culprit/file`
`black /path/to/culprit/file`

## Formatters

In order to homogenize the code base, we are including a couple of tools to help code formatting: `isort` for formatting imports and `black` for formatting code. The tools can be added with the following pip command:
```bash
pip install black isort
```

__Note: the convention is only being enforced for the following paths:__
- __`tests`__
- __`fugu/utils/validation.py`__
- __`fugu/simulators`__

__Note: exclude the following modules from automated formatting:__
- `fugu/backends`

You can run CI pipeline checks locally to check first:
```bash
isort --check --filter-files tests fugu/utils/validation.py fugu/simulators
black --check tests fugu/utils/validation.py fugu/simulators
```

The `filter-files` option ensures that files and directories are still skipped when specified into `isort` inputs.

There are various ways to automate these tools as part of your development: look up instructions for your text editor, IDE, etc. as well as Git pre-commit hooks.

_If you would like to exclude your source code from the auto-formatters, you can add the following at the top of your file(s):_
```python
"""
isort:skip_file
"""

# fmt: off
```

__Caution: if you are working with existing code that hasn't been formatted yet, please commit the updates from the formatting tools as a single commit before doing actual work and record the SHA as a new line in the file `.git-blame-ignore-revs`. This helps with more accurate information from the `git blame` command and prevent polluting the record with your username from the updates from the formatters. To configure `git` to use this file automatically, run the command `git config blame.ignoreRevsFile .git-blame-ignore-revs`.__

## Testing

Click [here](tests/README.md) for more information and instructions on Fugu's test suite.

## Branches

We suggest the following convention for naming branches: `username/##-branch-name`, where:
- `username`: your GitLab username
- `##`: issue number (can be omitted if branch is not tied to an issue)
- `branch-name`: a short descrition of the work

# Basic concepts

## Scaffold

The `Scaffold` object is a graph that contains bricks at each node.  In reality, the Scaffold is only responsible for the organization of bricks.  All functionality is held in the bricks themselves.


## Bricks

Each `Brick` represents one computational function.  Bricks are attached to a Scaffold.  Bricks have certain key properties:

- metadata:  A dictionary containing information such as the input and output sizes, circuit depth (if defined), and the types of codings.
- upported_codings:  A list of supported codings for this brick. A complete list of codings is avialable at input_coding_types .
- is_built:  A simple boolean saying whether or not the brick as been built
- name: A string representing the brick

### Brick.build(self, graph, metadata, complete_node, input_lists, input_codings)

This function forms the section of the graph corresponding to the brick.

Input parameters:
- graph: graph that is being built
- metadata: dictionary containing relevant dimensionality information
- control_nodes: A list of dictionaries of neurons that transmit control signals. A 'done' signal (Generally one from each input) is included in `control_nodes[i]['complete']`.
- input_lists: A list of lists of input neurons.  Each neuron is marked with a local index used for encodings.
- input_codings: A list of types of input codings.  See input_coding_types

Output:
A tuple (graph, metadata, complete_node, output_lists, output_codings)
- graph: Graph that is being built
- metadata: Dictionary containing relevant metadata information
- control_nodes: A dictionary of lists of neurons that transmists control information (see below)
- output_lists: A list of lists of output neurons.  Each neuron is marked with a local index used for encodings.
- output_codings: A list of types of codings.  See input_coding_types

### Details on `control_nodes`
We've seen through experience that it can be extremely helpful to have neurons relay control
information between bricks.  These neurons should be included in the `control_nodes`.  Regular
rules apply to these neurons (e.g. naming must be globally unique and indices are locally unique).

`control_nodes` is a *list* of *dictionaries*.  Each entry in the list corresponds with
an input (if calling `Brick.build`) or an output (if returning from `Brick.build`).

| Key | Required | Description |
| ------ | ------ | ------ |
| 'complete' | All Inputs/Outputs | A neuron that fires when a brick is done processing. |
| 'begin' | Temporally-coded Inputs/Outputs | A neuron that fires when a brick begins providing output. |
