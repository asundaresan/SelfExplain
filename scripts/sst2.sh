#!/bin/bash 

export DATA_FOLDER='data/SST-2-XLNet'
export TOKENIZER_NAME='xlnet-base-cased'
export MAX_LENGTH=5

# Creates jsonl files for train and dev
echo python bin/store_parse_trees.py --data_dir $DATA_FOLDER  --tokenizer_name $TOKENIZER_NAME

# Create concept store for dataset
# Since SST-2 already provides parsed output, easier to do it this way, for other datasets, need to adapt
echo python bin/build_concept_store.py -i $DATA_FOLDER/train_with_parse.json -o $DATA_FOLDER -m $TOKENIZER_NAME -l $MAX_LENGTH

echo python bin/train.py --dataset_basedir ${DATA_FOLDER} --lr 2e-5  --max_epochs 5 --gpus 1 --concept_store ${DATA_FOLDER}/concept_store.pt 

echo "--"
echo python bin/infer_model.py --concept_map ${DATA_FOLDER}/concept_idx.json \
  --dev_file ${DATA_FOLDER}/dev_with_parse.json \
  --ckpt lightning_logs/version_2/checkpoints/epoch\=2-step\=10524-val_acc_epoch\=0.9300.ckpt

echo python bin/self_explain_characterizer.py --concept_map ${DATA_FOLDER}/concept_idx.json \
  --tsv_filename ${DATA_FOLDER}/test.tsv \
  --checkpoint lightning_logs/version_2/checkpoints/epoch\=2-step\=10524-val_acc_epoch\=0.9300.ckpt
