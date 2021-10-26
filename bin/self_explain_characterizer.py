import logging
import json
import resource
import csv 
import os
import tqdm
import numpy as np
from argparse import ArgumentParser

from self_explain.plot_roc import plot_roc
from self_explain.json_util import load_json
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
    parser.add_argument('--model_conf', type=str, default=None, help="SE model to load")
    parser.add_argument('--tsv_filename', "-tf", type=str, default=None, help="dev/test TSV file to load")
    parser.add_argument('--number', "-n", type=int, default=0, help="Number of samples to process")
    parser.add_argument("--verbosity", "-v", action="count", default=0, help="Verbosity level")
    args = parser.parse_args()

    console_level = logging.WARN if args.verbosity == 0 else logging.INFO if args.verbosity == 1 else logging.DEBUG
    logging.basicConfig(level=console_level, format='[%(asctime)s %(levelname)s] %(message)s')

    kwargs = load_json(args.model_conf, keys=["concept_store",])
    ch = SelfExplainCharacterizer(**kwargs)

    data = load_tsv(args.tsv_filename)
    if args.number > 0 and args.number < len(data):
        data = data[:args.number]

    y_true = []
    y_pred = []
    for i, row in enumerate(tqdm.tqdm(data, desc="characterizer")):
        text = row["sentence"]
        #print(f"{i+1:04d}. '{text}'")
        try:
            prob, evidence = ch.process(text)
        except:
            print(f"failed to process '{text}'")
            continue
        result = dict(sentence=row["sentence"], label=row.get("label", None), prob=prob, evidence=evidence)
        y_true.append(row["label"])
        y_pred.append(prob)

    y_pred = np.array(y_pred).astype(float)
    y_true = np.array(y_true).astype(int)
    key = "_".join(os.path.splitext(args.tsv_filename)[0].split(os.path.sep)[-2:])
    save_dir = os.path.join("deploy", key)
    plot_roc(y_true, y_pred, save_dir=save_dir, key=key)
    np.savez("results.npz", y_true=y_true, y_pred=y_pred)
    print(f"y_true={y_true.sum()}/{y_true.size}")
    print(f"y_pred={np.round(y_pred).sum()}/{y_pred.size}")

