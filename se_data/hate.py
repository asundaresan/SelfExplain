import csv
import os
import collections
import logging 
import argparse

import tqdm

from .data import make_dataset

description = """Import hate-speech data into SE style data
"""

def load_csv(filename):
    data = list()
    with open(filename, "r") as handle:
        reader = csv.DictReader(handle)
        for row in tqdm.tqdm(reader, desc=f"loading {filename}"):
            data.append(row)
    print(f"loaded data ({len(data)}) from {filename}")
    return data


def import_wsf(filename: str, subfolder="all_files", max_contexts: int=0) -> dict:
    """ Import data from WSF
    """
    data = load_csv(filename)
    counters = dict()
    subfolder = os.path.join(os.path.dirname(filename), subfolder)
    se_data = {0: [], 1: []}
    label_map = {"hate": 1, "noHate": 0}
    for d in tqdm.tqdm(data, desc="importing"):
        sentence_filename = os.path.join(subfolder, f"{d['file_id']}.txt")
        with open(sentence_filename) as handle:
            sentences = handle.readlines()
        if len(sentences) == 1:
            sentence = sentences[0]
        else:
            print(f"found {len(sentences)} in {sentence_filename}, skipping.")
            continue
        for key in ['label', 'num_contexts', ]:
            if key not in counters:
                counters[key] = collections.Counter()
            counters[key].update([d[key],])
        num_contexts = int(d['num_contexts'])
        if num_contexts > max_contexts:
            # skip if num_contexts exceeds max_contexts
            continue
        key = d['label']
        try:
            label = label_map[key]
        except KeyError as e:
            logging.info(f"skipping label {key}: {e}")
            continue
        se_data[label].append(dict(sentence=sentence, label=label))
            
    for key, value in counters.items():
        print(f"- {key}: {value}")
    totals = {key: len(value) for key, value in se_data.items()}
    total = sum(totals.values())
    print(f"positive samples: {totals[1]}/{total}={totals[1]/total:.2f}")
    return se_data


def import_hsol(filename: str) -> dict:
    data = load_csv(filename)
    counters = dict()
    se_data = {0: [], 1: []}
    for d in data:
        label = 1 if int(d["class"])==0 else 0
        se_data[label].append(dict(sentence=d["tweet"], label=label))
        for key in ['count', 'hate_speech', 'offensive_language', 'neither', 'class',]:
            if key not in counters:
                counters[key] = collections.Counter()
            counters[key].update([d[key],])
    for key, value in counters.items():
        logging.info(f"{key}: {value}")
    totals = {key: len(value) for key, value in se_data.items()}
    total = sum(totals.values())
    print(f"positive samples: {totals[1]}/{total}={totals[1]/total:.2f}")
    return se_data





def import_hatespeech():
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("filename", type=str, help="Original data folder")
    parser.add_argument('--name', type=str, default="hsol", help="Hate speech dataset name")
    parser.add_argument('--save_dir', type=str, default=None, help="Directory to save TSV split files")
    parser.add_argument("--balance", "-b", action="store_true", help="Balance dataset by padding")
    parser.add_argument("--verbosity", "-v", action="count", default=0, help="Verbosity level")
    args = parser.parse_args()

    console_level = logging.WARN if args.verbosity == 0 else logging.INFO if args.verbosity == 1 else logging.DEBUG
    logging.basicConfig(level=console_level, format='[%(asctime)s %(levelname)s] %(message)s')

    save_dir = os.path.dirname(args.filename) if args.save_dir is None else args.save_dir
    if args.balance:
        save_dir = os.path.join(save_dir, "balanced")
        kwargs = dict(save_dir=save_dir, balance=True, pad=True,)
    else:
        kwargs = dict(save_dir=save_dir, balance=False,)
    if args.name == "hsol":
        se_data = import_hsol(args.filename)
    elif args.name == "wsf":
        se_data = import_wsf(args.filename)
    else:
        raise RuntimeError(f"Unknown dataset name: '{args.name}'")
    print(f"make_dataset({kwargs})")
    make_dataset(se_data, **kwargs)

