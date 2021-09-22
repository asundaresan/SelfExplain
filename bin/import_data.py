#!/usr/bin/env python 

import csv
import os
import collections
import logging 
import argparse

import tqdm

from self_explain.json_util import load_json
from self_explain.self_explain import convert_to_sentences

desc = """ Import data into SelfExplain style format """

def import_data(filenames, use_text=False) -> dict:
    se_data = dict()
    for filename in filenames:
        data = load_json(filename)
        for d in tqdm.tqdm(data, desc=f"importing {filename}"):
            label = int(d["score"])
            kwargs = dict(convert=True, label=label)
            all_sentences = list()
            if d.get("title") is not None:
                sentences = convert_to_sentences(d["title"], **kwargs)
                all_sentences.extend(sentences)
            if use_text:
                if d.get("text") is not None:
                    sentences = convert_to_sentences(d["text"], **kwargs)
                    all_sentences.extend(sentences)
            if not label in se_data:
                se_data[label] = []
            se_data[label].extend(all_sentences)
    return se_data


def make_balanced_dataset(data: dict, split=dict(train=0.90, dev=1.0), save_dir=None):
    """ Make balanced dataset from data
    """
    samples = {key: len(value) for key, value in data.items()}
    print(f"found {len(samples)} labels: {samples}")
    total = min(samples.values())
    start = 0
    for split_key, split_frac in split.items():
        end = int(split_frac*total)
        split_data = list()
        # for each label get the same amount of samples
        for key, value in data.items():
            print(f"  {split_key} <- {end-start} label '{key}'")
            split_data.extend(value[start:end])
        start = end
        counter = collections.Counter(s["label"] for s in split_data)
        print(f"{split_key}: {counter}")

        if save_dir is not None:
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            filename = os.path.join(save_dir, f"{split_key}.tsv")
            fieldnames = ["sentence", "label"]
            print(f"writing to {filename}: {len(split_data)} samples")
            with open(filename, "w") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
                writer.writeheader()
                for row in split_data:
                    writer.writerow(row)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument("filenames", nargs="+", type=str, help="JSON file data")
    parser.add_argument('--save_dir', type=str, default="data/fake", help="Directory to save TSV split files")
    parser.add_argument("--verbosity", "-v", action="count", default=0, help="Verbosity level")
    parser.add_argument("--debug", "-d", action="count", default=0, help="Debug level")
    args = parser.parse_args()

    console_level = logging.WARN if args.verbosity == 0 else logging.INFO if args.verbosity == 1 else logging.DEBUG
    logging.basicConfig(level=console_level, format='[%(asctime)s %(levelname)s] %(message)s')

    data = import_data(args.filenames)
    make_balanced_dataset(data, save_dir=args.save_dir)

