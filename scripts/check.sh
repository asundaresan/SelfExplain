#!/bin/bash 
# to run before pushing

echo running pylint 
python -m pylint -E --disable=no-name-in-module `dirname */__init__.py` tests bin
echo running pytest 
python -m pytest 
