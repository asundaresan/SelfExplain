#!/usr/bin/env python 

import os
import argparse
import json 
import numpy as np
import tqdm
import logging
from collections import Counter
import matplotlib.pyplot as plt


def histogram(text_lens: Counter, key=None, title="none", filename=None):
    max_len = max(text_lens)
    total = sum(text_lens.values())
    x = np.arange(max_len+1, dtype=int)
    y = [text_lens.get(i, 0) for i in x]
    fig = plt.figure(num=key)
    ax = fig.add_subplot(111)
    ax.bar(x, y)
    ax.set_xlabel(f"{key} length")
    ax.set_ylabel(f"number of {key}(s)")
    ax.set_xlim([0, max_len])
    ax2 = ax.twinx()
    y_cumsum = np.cumsum(y)*100/np.sum(y)
    ax2.plot(x, y_cumsum, label=f"num {key}s with len <=x")
    for l in [5, 10]:
        x_at_l = x[l-1:]
        y_at_l = y_cumsum[l]*np.ones_like(x_at_l)
        x_at_l[0] = x_at_l[1]
        y_at_l[0] = 0
        ax2.plot(x_at_l, y_at_l, "--", label=f"len({key}) <= {l}")
    ax2.legend()
    ax2.set_ylabel("perc under length")
    plt.suptitle(f"{title}, total {key}s={total}")
    if filename is not None: 
        print(f"saving to {filename}")
        fig.savefig(filename)


def get_matching_filepaths(folder, suffix=None):
    filepaths = []
    for root, dirnames, filenames in os.walk(folder):
        for f in filenames:
            if suffix is None or f.endswith(suffix):
                filepaths.append(os.path.join(root, f))
    return filepaths





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('filepaths', nargs="+", type=str, help="Files to parse")
    parser.add_argument('--phrase_labels', "-pl", nargs="+", default=["NP", "VP"], help="Phrase labels to use")
    parser.add_argument('--number', "-n", type=int, default=0, help="Number of sentences to analyze")
    parser.add_argument("--show", action="store_true", help="Show plot")
    parser.add_argument("--verbosity", "-v", action="count", default=0, help="Verbosity level")
    args = parser.parse_args()

    filenames = []
    for f in args.filepaths:
        if os.path.isfile(f):
            filenames.append(f)
        else:
            filepaths = get_matching_filepaths(f, suffix="train_with_parse.json")
            filenames.extend(filepaths)
    logging.info("\n".join(filenames))

    phrase_labels_info = ", ".join(args.phrase_labels)
    for filename in filenames:
        print("--")
        data = list()
        with open(filename, "r") as handle:
            for line in tqdm.tqdm(handle.readlines(), desc=f"reading {filename}"):
                data.append(json.loads(line))
                if len(data) > args.number and args.number > 0:
                    break
        print(f"loaded {filename} ({len(data)})")

        phrase_labels = Counter()
        sentence_lens = Counter()
        phrase_lens = Counter()
        phrases = set()
        for d in data: 
            sentence_len = len(d["sentence"].split())
            sentence_lens.update([sentence_len,])
            phrase_labels.update([leaf["phrase_label"] for leaf in d["parse_tree"]])
            for leaf in d["parse_tree"]:
                if leaf["phrase_label"] in args.phrase_labels:
                    phrase = leaf["phrase"].lower()
                    phrases.add(phrase)
                    phrase_len = len(phrase.split())
                    if phrase_len >= sentence_len:
                        logging.info(f"phrase_len={phrase_len} but sentence_len={sentence_len}\n\tphrase='{phrase}'\n\tsentence='{d['sentence']}'")
                    phrase_lens.update([phrase_len,])

        print(f"number of phrases ({phrase_labels_info}): {len(phrases)} from {len(data)} inputs")
        logging.info("phrase lengths: {phrase_labels}")

        # implicit structure of filename
        title = "/".join(filename.split(os.path.sep)[-3:-1])

        folder = os.path.join(os.path.dirname(filename), "analysis")
        if not os.path.exists(folder):
            os.makedirs(folder)

        output_filename = os.path.join(folder, "sentence.png")
        histogram(sentence_lens, key="sentence", title=title, filename=output_filename)

        output_filename = os.path.join(folder, "phrase.png")
        histogram(phrase_lens, key="phrase", title=title, filename=output_filename)

    if args.show:
        plt.show()
