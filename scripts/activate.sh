#!/bin/bash 

version=`python3 -c "import sys; version = '.'.join( map( str, sys.version_info[:2] ) ); print( version )"`
folder=venv/${HOSTNAME}_${version}
if [ $# -gt 0 ]; then 
  if [ -d $1 ]; then 
    echo "Using virtualenv ${1} instead of default ${folder}"
    folder=$1
  else
    echo "Folder does not exist: ${1}"
  fi
fi

if [[ ! -f ${folder}/bin/activate ]]; then
  python3 -m venv ${folder}
fi

source ${folder}/bin/activate
echo "Please run the following to update to latest: "
echo python -m pip install -U pip setuptools wheel

