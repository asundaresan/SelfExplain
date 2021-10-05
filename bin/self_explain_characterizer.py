import logging
import json
import resource
import csv 
import os
from argparse import ArgumentParser

from self_explain.self_explain import SelfExplainCharacterizer

def load_tsv(filename):
    data = []
    with open(filename) as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            data.append(row)
    info = ", ".join(data[0].keys()) if len(data) else "none"
    print(f"loaded {len(data)} samples from {filename}: {info}")
    return data


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--checkpoint_filename', type=str, default=None, help="Checkpoint to load")
    parser.add_argument('--concept_map_filename', type=str, default=None, help="Concept store file to load")
    parser.add_argument('--version', type=str, default="0.0.1", help="Model version number")
    parser.add_argument('--tsv_filename', type=str, default="data/SST-2-XLNet/test.tsv", help="dev/test TSV file to load")
    parser.add_argument('--number', "-n", type=int, default=0, help="Number of samples to process")
    parser.add_argument("--verbosity", "-v", action="count", default=0, help="Verbosity level")
    args = parser.parse_args()

    console_level = logging.WARN if args.verbosity == 0 else logging.INFO if args.verbosity == 1 else logging.DEBUG
    logging.basicConfig(level=console_level, format='[%(asctime)s %(levelname)s] %(message)s')

    data = load_tsv(args.tsv_filename)

    model_dir = os.path.join(os.path.expanduser('~'), ".malise", "models", "self_explain", args.version)
    checkpoint_filename = os.path.join(model_dir, "checkpoint.ckpt") if args.checkpoint_filename is None else args.checkpoint_filename
    concept_map_filename = os.path.join(model_dir, "concept_idx.json") if args.concept_map_filename is None else args.concept_map_filename

    kwargs = dict(checkpoint_filename=checkpoint_filename, concept_map_filename=concept_map_filename)
    ch = SelfExplainCharacterizer(**kwargs)

    if args.number > 0 and args.number < len(data):
        data = data[:args.number]

    results = list()
    for row in data:
        prob, evidence = ch.process(row["sentence"], convert=False)
        result = dict(sentence=row["sentence"], label=row.get("label", None), prob=prob, evidence=evidence)
        results.append(result)

    results_filename = os.path.splitext(args.tsv_filename)[0] + "_results.json"
    with open(results_filename, "w") as handle:
        json.dump(results, handle, indent=2)

