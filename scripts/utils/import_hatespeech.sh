#!/bin/bash 

SRC=/export/home/${HOSTNAME}1/data/semafor/hatespeech
DST=/export/home/${HOSTNAME}1/data/semafor/SE

echo python bin/util/import_hatespeech.py ${SRC}/hsol/labeled_data.csv --save_dir ${DST}/hsol --name hsol $*
echo python bin/util/import_hatespeech.py ${SRC}/wsf/annotations_metadata.csv --save_dir ${DST}/wsf --name wsf $*

