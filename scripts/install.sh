#!/bin/bash 
# from inside the virtual environment 

# update basic tools to latest version
pip install -U pip setuptools wheel
# install this package in editable mode
pip install -e .

