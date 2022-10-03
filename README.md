# Fugu
A python library for computational neural graphs.

# Install
```
git clone https://cee-gitlab.sandia.gov/nerl/Fugu.git
cd Fugu
pip install --upgrade .
```

An alternative way to install for development (using anaconda environments, python3)
```
conda create -n fugu anaconda
conda activate fugu
git clone git@cee-gitlab.sandia.gov:nerl/Fugu.git
cd Fugu
pip install -r requirements3.txt
pip install -e .
```

# Fugu Notes

Please see the provided example(s). 

## Scaffold

The `Scaffold` object is a graph that contains bricks at each node.  In reality, the Scaffold is only responsible for the organization of bricks.  All functionality is held in the bricks themselves.

### Scaffold.add_brick(self, brick_function, input_nodes=[-1], dimensionality = default_brick_dimensionality, name=None)

## Bricks

Each `Brick` represents one computational function.  Bricks are attached to a Scaffold.  Bricks have certain key properties:

- dimensionality:  A dictionary containing information such as the input and output sizes, circuit depth (if defined), and the types of codings.
- self.supported_codings:  A list of supported codings for this brick. A complete list of codings is avialable at input_coding_types .   
- is_built:  A simple boolean saying whether or not the brick as been built
- name: A string representing the brick

### Brick.build(self, graph, dimensionality, complete_node, input_lists, input_codings)

This function forms the section of the graph corresponding to the brick.

Input parameters:
- graph: graph that is being built
- dimensionality: dictionary containing relevant dimensionality information
- control_nodes: A list of dictionaries of neurons that transmit control signals. A 'done' signal (Generally one from each input) is included in `control_nodes[i]['complete']`.
- input_lists: A list of lists of input neurons.  Each neuron is marked with a local index used for encodings.
- input_codings: A list of types of input codings.  See input_coding_types

Output:
A tuple (graph, dimensionality, complete_node, output_lists, output_codings)
- graph: Graph that is being built
- dimensionality: Dictionary containing relevant dimensionality information
- complete_node: A list of neurons that transmists a 'done' signal (Generally one for each output)
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

## Documentation
Documentation is currently spread across several files.  We are working on including docstrings on all the classes and methods.

For now, you can check:
- This README.md
- docs.md
- Ipython notebook examples
- For details on ds, check the README.md of the ds_simulator package

To build the documentation, use pydocmd https://pypi.org/project/pydoc-markdown/ with a command similar to 
`pydocmd simple fugu++ > docs.md`

## Dependencies

- Numpy
- Scipy
- NetworkX
- Pandas
- Pytorch (for DS)


## Known Bugs and todos
New:
Jira Kanban Board - available to all members of wg-fugu
https://jira.sandia.gov/projects/FUGU/summary

Step 1: 
Join https://metagroup.sandia.gov/metagroups/wg-fugu
Step 2: Subscribe to Jira via Nile.  

Subscribing as a Jira User (required for CEE Jira access)
To subscribe to Jira User, go to Nile, CEE’s shopping cart portal.

    Navigate to Nile.
    Enter "Jira User" in the search field at the top center of the browser.
    Follow prompts to add the Jira User service to the cart.
    Complete the checkout. P/Ts will be required but not be charged but are collected for tracking purposes to better understand the programs at Sandia that are benefiting from CEE Jira.
    Access to CEE Jira should be available within 24 hours of subscribing. You will have access to any CEE Jira Project that has added you to their project Metagroup (wg-fugu) when you login.



- Combine and organize documentation
- ~~Inputs that take both pre-computed values and input spikes will not lay correctly.~~
- ~~Nodes only accept input from the first output of an input node.~~
- ~~`_create_ds_injection` only works on vector-shaped input~~
- ~~`_create_ds_injection` only works on input layers with 1 input (this may be okay)~~
- ~~Input handling needs to be re-written.  Currently relies on fragile order of nodes.  Additionally, we should support streaming data.~~
- Many checks are missing
- Mismatch between input and output layer namings.  Bricks should be able to be input and/or hidden and/or output layers.
- Maximum runtime should be determined by the depth of the graph.
- Delay brick needs to be updated as well as conversion bricks
- `Scaffold.resolve_timing` needs to be re-tested with new Delay brick
- ~~Dimensionliaty dictionary should be removed~~
- ~~Inputs/Outputs should be bundled as a list of dictionaries rather than parallel lists~~
- ~~Plotting functions likely do not and wll need to be re-tested~~ Vis is being re-written completely.
- Bricks in brick_1.py are likely broken and will need to be fixed.
- ~~Srideep: document fugu.py code bricks~~
- ~~Frances: documention around neural model Fugu is running (e.g. reference manual describing what is actually running)~~
- ~~Brad: focus on Fugu story (ICONS paper and Random Walk example)~~
- ~~Not Sam: Serious issues regarding how backend handles inputs~~
- Not Sam: Need scaffolds to be bricks
- ~~Not Sam: Redo how handle control nodes (Perphaps use dictionaries)~~
- ~~Expand outputs~~ 
- ~~Create backend class to handle backend simulators~~
- ~~Create installer for fugu~~

Link to internal Sharepoint for Development Next Steps - currently empty - consider using Jira instead:
- https://sandialabs-my.sharepoint.com/:w:/g/personal/iclane_sandia_gov/EY6DUQ8gXApIlc12uiBnKfUBDz_pj5adGUmYBnG6MLt5kQ?e=KC9Lzp
