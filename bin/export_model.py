import os
import logging
import json
import resource
import shutil
from argparse import ArgumentParser, Namespace

import torch 

from self_explain.model.SE_XLNet import SEXLNet
from self_explain.json_util import load_json, save_json
from self_explain import SelfExplainCharacterizer


def check_json(var: dict):
    """ Check if var is JSON serializable and pop values that are not!
    """
    keys = list(var.keys())
    for key in keys:
        try:
            json.dumps(var[key])
        except:
            print(f"* dropping: {key}: {var[key]}")
            var.pop(key)


""" Load from checkpoint and export model
"""

if __name__ == "__main__":
    #rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    #resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))
    base_dir = os.path.join(os.path.expanduser("~"), "malise", "models")

    parser = ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True, help="Checkpoint to load")
    parser.add_argument('--base_dir', type=str, default=base_dir, help="Output location for model")
    parser.add_argument('--name', type=str, default="self_explain_news", help="Name of model self_explain_news, self_explain_social")
    parser.add_argument('--save_dir', type=str, default=None, help="Output location")
    parser.add_argument('--version', type=str, default="0.0.1", help="Output location")
    parser.add_argument("--verbosity", "-v", action="count", default=0, help="Verbosity level")
    args = parser.parse_args()

    console_level = logging.WARN if args.verbosity == 0 else logging.INFO if args.verbosity == 1 else logging.DEBUG
    logging.basicConfig(level=console_level, format='[%(asctime)s %(levelname)s] %(message)s')

    print(f"loading from checkpoint: {args.checkpoint}")
    model = SEXLNet.load_from_checkpoint(args.checkpoint)
    dataset_basedir = model.hparams["dataset_basedir"]
    print(f"dataset_basedir: {dataset_basedir}")

    # folder to export to
    save_dir = os.path.join(args.base_dir, args.name, args.version) if args.save_dir is None else args.save_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    print(f"exporting model to {save_dir}")

    # export checkpoint 
    checkpoint_filename = os.path.join(save_dir, "checkpoint.ckpt")
    if True or not os.path.exists(checkpoint_filename):
        print(f"- copying to {checkpoint_filename}")
        shutil.copyfile(args.checkpoint, checkpoint_filename)
    else:
        print(f"- not copying to {checkpoint_filename} (file exists)")

    # export concept_map
    src = os.path.join(dataset_basedir, "concept_idx.json")
    dst = os.path.join(save_dir, os.path.basename(src))
    print(f"- copying to {dst}")
    shutil.copyfile(src, dst)
    concept_map_filename = dst


    # initialize model conf with checkpoint and concept_map
    conf = dict(checkpoint_filename=checkpoint_filename, concept_map_filename=concept_map_filename)

    # export concept_store and update kwargs and hparams 
    hparams = vars(model.hparams.hparams)
    check_json(hparams)

    # copy concept store from location in hparams
    checkpoint_kwargs = dict()
    for key in ["concept_store",]:
        src = model.hparams[key]
        dst = os.path.join(save_dir, os.path.basename(src))
        print(f"- copying to {dst}")
        shutil.copyfile(src, dst)
        checkpoint_kwargs[key] = os.path.abspath(dst)
        hparams[key] = checkpoint_kwargs[key]
    checkpoint_kwargs.update(dict(hparams=hparams))

    ## XXX not clear why concept_store is stored twice
    #concept_store = os.path.basename(model.hparams['concept_store'])
    #model.hparams['concept_store'] = concept_store
    #model.hparams.hparams.concept_store = concept_store
    #print(f"model.hparams['concept_store']={model.hparams['concept_store']}")
    #print(f"model.hparams.hparams.concept_store={model.hparams.hparams.concept_store}")

    # create configuration to load model from export dir
    conf.update(dict(checkpoint_kwargs=checkpoint_kwargs))
    conf_filename = os.path.join(save_dir, "model.json")
    conf_filename = os.path.abspath(conf_filename)
    print(f"- saving conf in {conf_filename}")
    save_json(conf, conf_filename, keys=["concept_store", ])

    # Load SelfExplainCharacterizer from exported model
    # change working directory so that relative paths are not valid
    print("--")
    os.chdir("/tmp")

    # load model conf, note that the concept_store and checkpoint_filename are expanded relative to os.path.dirname(conf_filename)
    conf = load_json(conf_filename, keys=["concept_store",])
    # get the checkpoint_filename
    checkpoint_filename = conf["checkpoint_filename"]
    if "checkpoint_kwargs" in conf:
        checkpoint_kwargs = conf["checkpoint_kwargs"]
        if "hparams" in checkpoint_kwargs:
            checkpoint_kwargs["hparams"] = Namespace(**checkpoint_kwargs["hparams"])
    print(f"loading {checkpoint_filename}: {checkpoint_kwargs}")

    # load model from exported checkpoint 
    # kwargs are specified to override parameters in model.hparams
    # XXX however concept_store is stored in 2 locations: 
    #   hparams["concept_store"]
    #   hparams.hparams.concept_store
    #model2 = SEXLNet.load_from_checkpoint(checkpoint_filename, **conf["checkpoint_kwargs"])
    se = SelfExplainCharacterizer(**conf)

