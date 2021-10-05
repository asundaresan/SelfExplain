import csv
import os
import collections
import logging 
import argparse

import tqdm

from .data import make_dataset

description = """Import ISOT data into SE style data
"""

def load_isot(filename: str, se_data: dict, label=None, use_text=True, use_title=False) -> dict:
    """ Import ISOT data into an SE compatible format
    """
    with open(filename, "r") as handle:
        reader = csv.DictReader(handle)

        for row in tqdm.tqdm(reader, desc=f"loading {filename}"):
            sentence = row.get("text")
            d = dict(sentence=sentence, label=label)
            if not label in se_data:
                se_data[label] = []
            se_data[label].append(d)


def import_isot():
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("folder", type=str, help="Original data folder")
    parser.add_argument('--save_dir', type=str, default=None, help="Directory to save TSV split files")
    parser.add_argument("--verbosity", "-v", action="count", default=0, help="Verbosity level")
    args = parser.parse_args()

    console_level = logging.WARN if args.verbosity == 0 else logging.INFO if args.verbosity == 1 else logging.DEBUG
    logging.basicConfig(level=console_level, format='[%(asctime)s %(levelname)s] %(message)s')

    true_filename = os.path.join(args.folder, "True.csv")
    fake_filename = os.path.join(args.folder, "Fake.csv")

    data = dict()
    load_isot(true_filename, data, label=0)
    load_isot(fake_filename, data, label=1)

    make_dataset(data, save_dir=args.save_dir, balance=True, pad=False)
