Documentation
=============

Documentation for tabular_trees is generated using [Sphinx](https://www.sphinx-doc.org/).

Build
-----

To build the documentation locally, first run the following from the root of the project to install the documentation packages

```
pip install flit
flit install --deps develop
```

Then run the following to build the documentation

```
cd docs
make html
```
