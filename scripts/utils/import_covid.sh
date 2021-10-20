#!/bin/bash 

DATA=/export/home/`hostname -s`1/data/semafor

echo python bin/util/import_covid.py ${DATA}/covid/*.json.gz --save_dir ${DATA}/SE/covid $*
