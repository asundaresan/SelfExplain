#!/bin/bash 

python bin/infer_model.py --concept_map data/SST-2-XLNet/concept_idx.json \
  --dev_file data/SST-2-XLNet/dev_with_parse.json \
  --ckpt lightning_logs/version_2/checkpoints/epoch\=2-step\=10524-val_acc_epoch\=0.9300.ckpt
