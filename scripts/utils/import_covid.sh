#!/bin/bash 

DATA=/export/home/${HOSTNAME}1/data/semafor/covid

python bin/util/import_covid.py ${DATA}/data/*.json.gz $*
