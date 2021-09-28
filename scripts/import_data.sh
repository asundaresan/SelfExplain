#!/bin/bash 

DATA=/export/home/${HOSTNAME}1/data/semafor/fake

python bin/import_data.py ${DATA}/* $*
