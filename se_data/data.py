import csv
import os
import collections
import logging 
import gzip


def make_dataset(class_data: dict, split=dict(train=0.80, dev=0.1, test=0.1), balance=True, pad=True, save_dir=None, compress=True):
    """ Make balanced dataset from class_data. 
    Class data is a dict that consists of two keys: 0 and 1
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
        logging.info(f"class '{key}': {len(value)} samples")
        for split_key, split_frac in split.items():
            split_len = int(split_frac*len(value))
            split_total = int(split_frac*total) if total is not None else None
            logging.info(f"  need to get {split_total} from {split_len}")
            if split_total is None:
                # balance=False
                end = start+split_len
                split_data[split_key].extend(value[start:end])
                info = f"{start}:{end} ({split_total} from {split_len} available)"
            elif split_len >= split_total:
                # balance=True, pad=False
                end = start+split_total
                split_data[split_key].extend(value[start:end])
                info = f"{start}:{end} ({split_total} from {split_len} available)"
            else:
                # balance=True, pad=True
                # padding will repeat low population classes wholly (not partially)
                # it will be strictly lesser than 
                end = start+split_len
                repeat = split_total//split_len
                info = f"{start}:{end} x {repeat} ({split_total} from {split_len} available)"
                for _ in range(repeat):
                    split_data[split_key].extend(value[start:end])
            logging.info(f"  {split_key} <- {info}")
            start = end
                    
    for split_key, data in split_data.items():
        counter = collections.Counter(s["label"] for s in data)
        logging.info(f"{split_key}: {counter}")

        if save_dir is not None:
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            extension = "tsv.gz" if compress else "tsv"
            filename = os.path.join(save_dir, f"{split_key}.{extension}")
            fieldnames = ["sentence", "label"]
            print(f"writing to {filename}: {len(data)} samples")
            handle = gzip.open(filename, "wt") if compress else open(filename, "w")
            try:
                writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
                writer.writeheader()
                for row in data:
                    writer.writerow(row)
            finally:
                handle.close()


