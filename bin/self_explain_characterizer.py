import logging
import json
import resource
import csv 
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
    #rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    #resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))
    parser = ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default="model/checkpoint.ckpt", help="Checkpoint to load")
    parser.add_argument('--concept_map_filename', type=str, default="model/concept_idx.json", help="Concept store file to load")
    parser.add_argument('--tsv_filename', type=str, default="data/SST-2-XLNet/test.tsv", help="dev/test TSV file to load")
    parser.add_argument('--number', "-n", type=int, default=0, help="Number of samples to process")
    parser.add_argument("--verbosity", "-v", action="count", default=0, help="Verbosity level")
    args = parser.parse_args()

    console_level = logging.WARN if args.verbosity == 0 else logging.INFO if args.verbosity == 1 else logging.DEBUG
    logging.basicConfig(level=console_level, format='[%(asctime)s %(levelname)s] %(message)s')

    data = load_tsv(args.tsv_filename)

    kwargs = dict(checkpoint=args.checkpoint, concept_map_filename=args.concept_map_filename)
    ch = SelfExplainCharacterizer(**kwargs)

    if args.number > 0 and args.number < len(data):
        data = data[:args.number]

    for row in data:
        result = ch.process(row["sentence"], convert=False)

