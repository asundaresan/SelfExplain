#!/bin/bash 

if [[ -z $1 ]]; then 
  folder=/export/home/`hostname -s`1/data/semafor/SE
else
  folder=$1
fi
for events in `find ${folder} -iname events.out.\*`; do 
  logdir=`dirname $events`
  echo tensorboard --logdir=$logdir --host `hostname -s`
done | sort | tail -n 10
