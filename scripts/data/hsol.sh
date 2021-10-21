#!/bin/bash 

export EXPERIMENT="hsol"
export DATA_FOLDER="data/semafor/SE/${EXPERIMENT}/balanced"
export TOKENIZER_NAME='xlnet-base-cased'
export MAX_LENGTH=10

echo export CUDA_VISIBLE_DEVICES=1

# Creates jsonl files for train and dev
echo python bin/store_parse_trees.py --data_dir $DATA_FOLDER  --tokenizer_name $TOKENIZER_NAME

# Create concept store for dataset
# Since SST-2 already provides parsed output, easier to do it this way, for other datasets, need to adapt
echo python bin/build_concept_store.py -i $DATA_FOLDER/train_with_parse.json -o $DATA_FOLDER -m $TOKENIZER_NAME -l $MAX_LENGTH

echo python bin/train.py --dataset_basedir ${DATA_FOLDER} --lr 2e-5  --max_epochs 5 --gpus 1 --concept_store ${DATA_FOLDER}/concept_store.pt --default_root_dir logs/${EXPERIMENT}

echo "--"
echo find logs/${EXPERIMENT} -iname epoch\*.ckpt 

echo python bin/infer_model.py --concept_map ${DATA_FOLDER}/concept_idx.json \
  --dataset_basedir ${DATA_FOLDER} \
  --ckpt 

echo python bin/self_explain_characterizer.py --concept_map ${DATA_FOLDER}/concept_idx.json \
  --tsv_filename ${DATA_FOLDER}/test.tsv \
  --checkpoint

echo "--"
MODEL_FOLDER=~/malise/models/self_explain/0.0.1
echo python bin/self_explain_characterizer.py --concept_map ${MODEL_FOLDER}/concept_idx.json \
  --tsv_filename ${DATA_FOLDER}/test.tsv \
  --checkpoint 
