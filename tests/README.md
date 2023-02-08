# Fugu Test Suite

## Get Started

```bash
pip install -U pytest coverage
pytest
```

__Note that you can run the legacy test suite pointing PyTest to the old test suite, i.e. `pytest test`__

### Coverage

```bash
coverage run -m pytest tests --disable-warnings
coverage report -m
```

And to generate a HTML report
```bash
coverage html
```
Then open `htmlcov/index.html` in a browser for a more in-depth analysis of the coverage report.

## Setup

The new test suite is organized into directories for the type of testing they represent.
- [unit](unit) - set of unit tests