# RADIomic Spatial TexturAl descripTor: Python implemntation

RADISTAT is a radiomic feature descriptor that characterizes the spatial and
textural heterogeneity of a given radiomic feature.

This repository is a library which calculates the two metrics described in the
RADISTAT paper.

<!-- TODO expand description -->

## Getting Started

Right now, to use this library, you'll need to clone this repository and
install Python dependencies, preferably in a virtual environment to keep things
clean.

There are two ways to do this:

### Poetry

If you have [Poetry](https://python-poetry.org) installed, run
```shell
poetry install
```

This will create a virtual environment with all requirements installed for you.
To run something inside this environment, just use
```shell
poetry run [commmand]
```
or to spawn a shell inside the environment, run
```shell
poetry shell
```

### Virtualenv

Requirements are also given in `requirements.txt`.

Given that you have Python 3 installed, run
```shell
python3 -m venv venv/
./venv/bin/activate
pip install -r requirements.txt
```

## Usage

Currently the library is only available as a Python module.
See the provided [Jupyter notebook](./demo_radistat.ipynb) for an example of how to use this library.
