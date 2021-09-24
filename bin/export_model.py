import os
import logging
import json
import resource
import shutil
from argparse import ArgumentParser

import torch 

from self_explain.model.SE_XLNet import SEXLNet
from self_explain.json_util import load_json, save_json

""" Load from checkpoint and export model
"""

if __name__ == "__main__":
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))

    parser = ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True, help="Checkpoint to load")
    parser.add_argument('--output', type=str, default=None, help="Output location")
    parser.add_argument('--version', type=str, default="0.0.1", help="Output location")
    parser.add_argument("--verbosity", "-v", action="count", default=0, help="Verbosity level")
    args = parser.parse_args()

    console_level = logging.WARN if args.verbosity == 0 else logging.INFO if args.verbosity == 1 else logging.DEBUG
    logging.basicConfig(level=console_level, format='[%(asctime)s %(levelname)s] %(message)s')

    print(f"loading from checkpoint: {args.checkpoint}")
    model = SEXLNet.load_from_checkpoint(args.checkpoint)
    # XXX not clear why concept_store is stored twice
    print("model.hparams['concept_store']={model.hparams['concept_store']}")
    print("model.hparams.hparams.concept_store={model.hparams.hparams.concept_store}")
    
    # folder to export to
    save_dir = os.path.join("models", "self_explain", args.version)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # location to save the checkpoint file 
    checkpoint_filename = os.path.join(save_dir, "checkpoint.ckpt")
    print(f"exporting model to {save_dir}")
    if not os.path.exists(checkpoint_filename):
        print(f"- copying to {checkpoint_filename}")
        shutil.copyfile(args.checkpoint, checkpoint_filename)
    else:
        print(f"- not copying to {checkpoint_filename} (file exists)")

    # copy concept store from location in hparams
    kwargs = dict()
    for key in ["concept_store",]:
        src = model.hparams[key]
        dst = os.path.join(save_dir, os.path.basename(src))
        print(f"- copying to {dst}")
        shutil.copyfile(src, dst)
        kwargs[key] = os.path.abspath(dst)
    # create configuration to load model from export dir
    conf = dict(checkpoint_filename=checkpoint_filename, kwargs=kwargs)
    conf_filename = os.path.join(save_dir, "model.json")
    conf_filename = os.path.abspath(conf_filename)
    print(f"- saving conf in {conf_filename}")
    save_json(conf, conf_filename, keys=["concept_store", ])

    # change working directory so that relative paths are not valid
    os.chdir("/tmp")

    # load model conf, note that the concept_store and checkpoint_filename are expanded relative to os.path.dirname(conf_filename)
    conf = load_json(conf_filename, keys=["concept_store",])
    checkpoint_filename = conf.pop("checkpoint_filename")
    print(f"loading {checkpoint_filename}: {conf['kwargs']}")

    # load model from exported checkpoint 
    # kwargs are specified to override parameters in model.hparams
    # XXX however concept_store is stored in 2 locations in model.hparams 
    # model.hparams["concept_store"] and model.hparams.hparams.concept_store
    model2 = SEXLNet.load_from_checkpoint(checkpoint_filename, **conf["kwargs"])

    # saving to torchscript does not work
    #torch.jit.save(model.to_torchscript(), filename)

