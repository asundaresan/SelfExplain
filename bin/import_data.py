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


def make_dataset(class_data: dict, split=dict(train=0.80, dev=0.1, test=0.1), balance=True, pad=True, save_dir=None):
    """ Make balanced dataset from class_data
    Args: 
        class_data (dict): data with samples stored as a list for each class (key)
        split (dict): how to split data for training, dev (validation) and test
        balance (bool): should the data be balanced, i.e. all classes should have the same amount of data
        pad (bool): should the data be padded to make balanced dataset?
    """
    samples = {key: len(value) for key, value in class_data.items()}
    print(f"found {len(samples)} labels: {samples}")
    # set total to None if dataset does not have to be balanced else total samples for each class
    if balance:
        total = max(samples.values()) if pad else min(samples.values())
    else:
        total = None
    # create split_data 
    split_data = {key: [] for key in split}
    # for each label get the same amount of samples
    for key, value in class_data.items():
        start = 0
        class_total = len(value)
        print(f"class '{key}': {len(value)} samples")
        for split_key, split_frac in split.items():
            split_len = int(split_frac*len(value))
            split_total = int(split_frac*total) if total is not None else None
            print(f"-- need to get {split_total} from {split_len}")
            if split_total is None:
                # balance=False
                end = start+split_len
                split_data[split_key].extend(value[start:end])
                info = f"{start}:{end}"
            elif split_len >= split_total:
                # balance=True, pad=False
                end = start+split_total
                split_data[split_key].extend(value[start:end])
                info = f"{start}:{end}"
            else:
                # balance=True, pad=True
                # padding will repeat low population classes wholly (not partially)
                # it will be strictly lesser than 
                end = start+split_len
                repeat = split_total//split_len
                info = f"{start}:{end} x {repeat}"
                for _ in range(repeat):
                    split_data[split_key].extend(value[start:end])
            print(f"  {split_key} <- {info}")
            start = end
                    
    for key, data in split_data.items():
        counter = collections.Counter(s["label"] for s in data)
        print(f"{split_key}: {counter}")

        if save_dir is not None:
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            filename = os.path.join(save_dir, f"{split_key}.tsv")
            fieldnames = ["sentence", "label"]
            print(f"writing to {filename}: {len(data)} samples")
            with open(filename, "w") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
                writer.writeheader()
                for row in data:
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
    make_dataset(data, save_dir=args.save_dir, balance=True, pad=True)

