#!/bin/bash 

DATA=/export/home/${HOSTNAME}1/data/semafor/hatespeech

python bin/util/import_hatespeech.py ${DATA}/hsol/labeled_data.csv --name hsol $*
python bin/util/import_hatespeech.py ${DATA}/wsf/annotations_metadata.csv --name wsf $*

