# Fugu Test Suite

## Get Started

```bash
pip install -U pytest==7.4.4 coverage==7.4.1
pytest
```

__Note that you can run the legacy test suite pointing PyTest to the old test suite, i.e. `pytest legacy_tests`__

### Coverage

```bash
coverage run -m pytest
coverage report -m
```

And to generate a HTML report
```bash
coverage html
```
Then open `htmlcov/index.html` in a browser for a more in-depth analysis of the coverage report.

Additionally, the script `coverage.sh` can be ran to update your local coverage report, if needed.

## Setup

The new test suite is organized into directories for the type of testing they represent.
- [unit](unit) - set of unit tests