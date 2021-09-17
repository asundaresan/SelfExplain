# SelfExplain Framework

The code for the SelfExplain framework (https://arxiv.org/abs/2103.12279) 

Currently, this repo supports SelfExplain-XLNet version for SST-2 dataset. The other datasets and models shown in the paper will be updated soon. Note that this is a fork and the original repository is here: http://github.com/dheerajrajagopal/SelfExplain.git


## Set up virtualenv and install dependencies 

To activate a virtualenv, run the following command. 

```shell
source scripts/activate.sh
```

To install the dependencies and the `self_explain` package inside the virtualenv, run the following command.

```shell
source scripts/install.sh
```

## Preprocessing

Data for preprocessing available in `data/` folder

Do the following for installing the parser

```shell
python -c "import benepar; benepar.download('benepar_en3')"
```

To store the parse tree and build the concept store for SST-2-XLNet, run the following commands.

```shell
python bin/store_parse_trees.py --data_dir data/SST-2-XLNet --tokenizer_name xlnet-base-cased
python bin/build_concept_store.py -i data/SST-2-XLNet/train_with_parse.json -o data/SST-2-XLNet -m xlnet-base-cased -l 5
```
To store the parse tree and build the concept store for COVID, run the following commands.

```shell
python bin/store_parse_trees.py --data_dir data/covid --tokenizer_name xlnet-base-cased
python bin/build_concept_store.py -i data/covid/train_with_parse.json -o data/SST-2-XLNet -m xlnet-base-cased -l 5
```

## Training

(In Progress)

```shell
sh scripts/run_self_explain.sh
```
## Generation (Inference)

(In Progress)

```sh
 python model/infer_model.py
        --ckpt $PATH_TO_BEST_DEV_CHECKPOINT \
        --concept_map $DATA_FOLDER/concept_idx.json \ 
        --batch_size $BS \
        --paths_output_loc $PATH_TO_OUTPUT_PREDS \
        --dev_file $PATH_TO_DEV_FILE
 ```

## Demo 

Coming Soon ... 

## Citation 

```
@misc{rajagopal2021selfexplain,
      title={SelfExplain: A Self-Explaining Architecture for Neural Text Classifiers}, 
      author={Dheeraj Rajagopal and Vidhisha Balachandran and Eduard Hovy and Yulia Tsvetkov},
      year={2021},
      eprint={2103.12279},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
