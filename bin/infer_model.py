import os
import logging
import json
import resource
from argparse import ArgumentParser

from self_explain.model.infer_model import evaluate, load_model, load_concept_map

if __name__ == "__main__":
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))

    parser = ArgumentParser()
    parser.add_argument('--ckpt', type=str, required=True, help="Checkpoint to load")
    parser.add_argument('--concept_map', type=str, required=True, help="Concept store file to load")
    parser.add_argument("--dataset_basedir", help="Base directory where the dataset is located.", type=str)
    parser.add_argument('--batch_size', type=int, default=16, help="Batch size to use")
    parser.add_argument("--verbosity", "-v", action="count", default=0, help="Verbosity level")
    args = parser.parse_args()

    console_level = logging.WARN if args.verbosity == 0 else logging.INFO if args.verbosity == 1 else logging.DEBUG
    logging.basicConfig(level=console_level, format='[%(asctime)s %(levelname)s] %(message)s')

    print(f"loading checkpoint from: {args.ckpt}")
    model, trainer, dm = load_model(args.ckpt, batch_size=args.batch_size)

    print(f"loading concept_map from: {args.concept_map}")
    concept_map = load_concept_map(args.concept_map)

    splits = {"test": dict(filename="test_with_parse.json", dataloader=dm.test_dataloader()),
        "validation": dict(filename="dev_with_parse.json", dataloader=dm.val_dataloader()),
        }


    for key, value in splits.items():
        # get validation dataloader
        dataloader = value["dataloader"]
        dev_file = os.path.join(args.dataset_basedir, value["filename"])

        # save output to dev_file + "_output.tsv" if not specified
        paths_output_loc = os.path.splitext(value["filename"])[0] + "_output.tsv"
        paths_output_loc = os.path.join(args.dataset_basedir, "results", paths_output_loc)

        evaluate(model, dataloader, concept_map=concept_map, dev_file=dev_file, paths_output_loc=paths_output_loc)

