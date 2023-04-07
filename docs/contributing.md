layout: page title: "Contributing" permalink: /contributing

# Contributing to Fugu

Fugu's designs mean there are several ways to add new components (e.g. bricks and backends) to Fugu.  Over time, new features and components will be added to Fugu itself, but we expect that Fugu will only include a base collection of bricks and backends.  

For those wanting to build extensions, we suggest extending the relevant class and releasing a separate package as a collection of classes. Most likely the relevant class for a new brick is `fugu.bricks.bricks.Brick`, and for a backend it is `fugu.backends.backend.Backend`. We suggest including a required Fugu verison in `setup.py`. 

As an example, suppose someone create a collection of bricks focused on signal processing.  The developer of those bricks will need Fugu installed and should extend `fugu.bricks.bricks.Brick` in their code.  They can then release a package, called for example SignalBricks, of the bricks citing Fugu as a dependency.  Users who are interested in SignalBricks can install this package, and base Fugu should automatically be installed as a dependency.  Alternatively, users can install Fugu followed by installing SignalBricks; the end result is the same.  The user would then import Fugu modules from Fugu and SignalBricks modules from SignalBricks, but they should work interchangably.

We hope that well-adopted extensions will be incorporated into base Fugu over time.

The advantages of this method are:
- Contributors can release extensions without any burdens from base Fugu
- Fugu still contains a usesable set of bricks and backends
- Users only need to download the components they want to use

The disadvantages of this method are:
- It may be difficult for a user to find what bricks are available
- Users will need to keep track of which bricks are from which sources
- Potential for versioning challenges

In the future, we can investigate methods to overcome these disadvantages, such as an extensions library, contrib package, or install extras.
