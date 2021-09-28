#!/bin/bash 

ls lightning_logs/version_*/checkpoints/epoch* -t | sort | tail 

checkpoint=`ls lightning_logs/version_*/checkpoints/epoch* -t | tail -1`
echo python bin/export_model.py --checkpoint ${checkpoint}
