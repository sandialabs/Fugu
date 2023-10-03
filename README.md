# Fugu
A python library for computational neural graphs.

_Note: the `master` branch has been renamed to `main`. We've kept an old version in case it would break any local branches, but on your next git push type the following command to have your origin branch renamed on your local machine:_
```bash
git push origin HEAD
```

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
git clone https://cee-gitlab.sandia.gov/nerl/Fugu.git
cd Fugu
pip install -r requirements.txt
pip install --upgrade .
```

## Using Conda
Let's first install the requirements for Fugu.
```bash
git clone https://cee-gitlab.sandia.gov/nerl/Fugu.git
cd Fugu
conda env create -f fugu_conda_environment.yml
conda activate fugu
conda develop $PWD
```

Next we will install the nxsdk package. But before we can do this, we will need to manually modify the contents inside nxsdk tar file.

1. Untar the nxsdk-\<version\>.tar.gz package to `/PATH/TO/nxsdk-<version>`
2. Fix the version format so that it is PEP400 compatible.
    - Modify the `/PATH/TO/nxsdk-<version>/nxsdk/version.py`
        - Need to replace the '-' after the third number to a '+'.
        - AA.BB.CC-\<STUFF\> ----> AA.BB.CC+\<STUFF\>
3. Replace nxsdk requirements.txt (`/PATH/TO/nxsdk-<version>/requirements.txt`) file with
```bash
attrdict>=2.0.1
numpy==1.15.*
pandas>=1.0.*
matplotlib>=2.2.2
imageio>=2.6.1
scikit-image>=0.14.2
scikit-learn>=0.19.2
jinja2>=2.10
coloredlogs>=10.0
grpcio>=1.19.0
protobuf==3.19.*
grpcio_tools>=1.19.0
memory_profiler>=0.55
bitstring>=3.1.6
```

Then run
```bash
conda activate fugu
python -m pip install /PATH/TO/nxsdk-<version>/
conda develop /PATH/TO/nxsdk-apps/nxsdk_modules
```

# Documentation
Documentation is currently spread across several files and directories.  We are working on including docstrings on all the classes and methods.

For now, you can check:
- http://nerl.cee-gitlab.lan/Fugu/
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
