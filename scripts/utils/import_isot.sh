#!/bin/bash 

DATA=/export/home/`hostname -s`1/data/semafor
python3 bin/util/import_isot.py ${DATA}/ISOT --save_dir ${DATA}/SE/ISOT

