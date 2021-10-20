#!/usr/bin/env python 

import csv
import os
import collections
import logging 
import argparse
import json 
import gzip

import tqdm

from se_data.data import make_dataset

desc = """ Import COVID data into SelfExplain style format """

def import_covid(filenames, use_text=False) -> dict:
    se_data = {0: [], 1: []}
    for filename in filenames:
        print(f"reading from {filename}")
        with gzip.open(filename, "rt") as handle:
            data = json.load(handle)
        for d in tqdm.tqdm(data, desc=f"importing {filename}"):
            label = int(d["score"])
            kwargs = dict(convert=True, label=label)
            if d.get("title") is not None:
                se_data[label].append(dict(sentence=d["title"], label=label))
    totals = {key: len(value) for key, value in se_data.items()}
    total = sum(totals.values())
    print(f"positive samples: {totals[1]}/{total}={totals[1]/total:.2f}")
    return se_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument("filenames", nargs="+", type=str, help="JSON file data")
    parser.add_argument('--save_dir', type=str, default=None, help="Directory to save TSV split files")
    parser.add_argument("--balance", "-b", action="store_true", default=False, help="Balance dataset by padding")
    parser.add_argument("--verbosity", "-v", action="count", default=0, help="Verbosity level")
    parser.add_argument("--debug", "-d", action="count", default=0, help="Debug level")
    args = parser.parse_args()

    console_level = logging.WARN if args.verbosity == 0 else logging.INFO if args.verbosity == 1 else logging.DEBUG
    logging.basicConfig(level=console_level, format='[%(asctime)s %(levelname)s] %(message)s')


    save_dir = os.path.dirname(args.filenames[0]) if args.save_dir is None else args.save_dir
    if args.balance: 
        save_dir = os.path.join(save_dir, "balanced")
        kwargs = dict(save_dir=save_dir, balance=True, pad=True)
    else:
        kwargs = dict(save_dir=save_dir, balance=False)

    se_data = import_covid(args.filenames)
    make_dataset(se_data, **kwargs)

