import csv
import os
import collections
import logging 
import argparse

import tqdm

from .data import make_dataset

description = """Import ISOT data into SE style data
"""

def clean_text(sentence):
    val = sentence.split(" - ")
    if len(val[0].split()) < 10:
        clean_text.sources.update(val[0:1])
        sentence = " - ".join(val[1:])
    return sentence
clean_text.sources = collections.Counter()


def load_isot(filename: str, se_data: dict, label=None, use_text=True, use_title=False) -> None:
    """ Import ISOT data into an SE compatible format
    """
    with open(filename, "r") as handle:
        reader = csv.DictReader(handle)

        for row in tqdm.tqdm(reader, desc=f"loading {filename}"):
            sentence = row.get("text")
            sentence = clean_text(sentence)
            d = dict(sentence=sentence, label=label)
            if not label in se_data:
                se_data[label] = []
            se_data[label].append(d)


def import_isot():
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("folder", type=str, help="Original data folder")
    parser.add_argument('--save_dir', type=str, default=None, help="Directory to save TSV split files")
    parser.add_argument("--balance", "-b", action="store_true", help="Balance dataset by padding")
    parser.add_argument("--verbosity", "-v", action="count", default=0, help="Verbosity level")
    args = parser.parse_args()

    console_level = logging.WARN if args.verbosity == 0 else logging.INFO if args.verbosity == 1 else logging.DEBUG
    logging.basicConfig(level=console_level, format='[%(asctime)s %(levelname)s] %(message)s')

    true_filename = os.path.join(args.folder, "True.csv")
    fake_filename = os.path.join(args.folder, "Fake.csv")

    se_data = dict()
    load_isot(true_filename, se_data, label=0)
    logging.info(f"sources={clean_text.sources}")
    load_isot(fake_filename, se_data, label=1)

    totals = {key: len(value) for key, value in se_data.items()}
    total = sum(totals.values())
    print(f"positive samples: {totals[1]}/{total}={totals[1]/total:.2f}")

    save_dir = os.path.dirname(args.filename) if args.save_dir is None else args.save_dir
    kwargs = dict(save_dir=save_dir)
    if args.balance:
        kwargs.update(dict(balance=True, pad=True,)) 
    else:
        kwargs.update(dict(balance=False,)) 

    make_dataset(se_data, save_dir=args.save_dir, balance=True, pad=False)
