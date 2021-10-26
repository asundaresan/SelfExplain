""" JSON utilities 
""" 
import json 
import gzip 
import logging 
import os
import copy
import numpy as np

def key_match(key, suffixes=[], prefixes=[], keys=[]):
    if key in keys:
        return True
    for suffix in suffixes:
        if key.endswith(suffix):
            return True
    for prefix in prefixes:
        if key.startswith(prefix):
            return True
    return False



def join_relative_path(data, folder, keys=[], suffixes=["filename", "filepattern",], prefixes=[]):
    """ Iterate recursively through data and whenever there is a key match 
    (i.e. key in keys or key.endswith(suffix) or key.startswith(prefix))
    and type(value) is str, join the path to folder to get full path.
    """
    if isinstance(data, dict):
        for key, value in data.items():
            if key_match(key, keys=keys, suffixes=suffixes, prefixes=prefixes):
                logging.log(1, "matched {key}: {value}")
                if isinstance(value, str): 
                    data[key] = os.path.join(folder, value)
                elif isinstance(value, list) and all(isinstance(v2, str) for v2 in value): 
                    data[key] = list(os.path.join(folder, v2) for v2 in value)
            join_relative_path(value, folder, keys=keys)
    elif isinstance(data, list):
        for d in data:
            join_relative_path(d, folder, keys=keys)


def make_relative_path(data, folder, keys=[], prefixes=[], suffixes=["filename", "filepattern",] ):
    """ Iterate recursively through data and whenever a (key, value) pair is such that the key
    matches (keys, suffixes, prefixes) and value is str or list(str), then compute path(s)
    relative to folder. 
    """
    kwargs = dict(keys=keys, suffixes=suffixes, prefixes=prefixes)
    if isinstance(data, dict):
        for k, v in data.items():
            if key_match(k, **kwargs):
                if isinstance(v, str):
                    data[k] = os.path.relpath(v, folder)
                    logging.debug(f"setting {k}: {v} -> {data[k]}")
                elif isinstance(v, list) and all(isinstance(v2, str) for v2 in v):
                    data[k] = list(os.path.relpath(v2, folder) for v2 in v)
                    logging.debug(f"setting {k}: {v} -> {data[k]}")
            make_relative_path(v, folder, **kwargs)
    elif isinstance(data, list):
        for d in data:
            make_relative_path(d, folder, **kwargs)



def dict_to_numpy(data):
    """ Iterate recursively through data and replace numpy arrays { k: v }  with { k.numpy: v.tolist() }
    """
    def dict_to_numpy_(data):
        if isinstance(data, dict):
            keys = list(data.keys())
            for k in keys:
                if k.endswith(".__ndarray__"):
                    v = data.pop(k)
                    k2 = k.replace(".__ndarray__", "")
                    data[k2] = np.array(v["data"], dtype=v["dtype"])
                else:
                    dict_to_numpy_(data[k])
        elif isinstance(data, list):
            for d in data:
                dict_to_numpy_(d)
    data2 = copy.deepcopy(data)
    dict_to_numpy_(data2)
    return data2


def numpy_to_dict(data):
    """ Iterate recursively through data and replace numpy arrays { k: v }  with { k.numpy: v.tolist() }
    """
    def numpy_to_dict_(data):
        if isinstance(data, dict):
            keys = list(data.keys())
            for k in keys:
                v = data[k]
                if isinstance(v, np.ndarray):
                    k2 = "%s.__ndarray__" % k
                    v = data.pop(k)
                    data[k2] = dict(data=v.tolist(), dtype=v.dtype.name)
                else:
                    numpy_to_dict_(v)
        elif isinstance(data, list):
            for d in data:
                numpy_to_dict_(d)
    data2 = copy.deepcopy(data)
    numpy_to_dict_(data2)
    return data2


def load_json(filename, folder=None, **kwargs):
    """ Load data from file 
    """
    folder = os.path.dirname(filename) if folder is None else folder
    if filename.endswith(".json"):
        with open(filename, "r") as handle:
            data = json.load(handle)
    elif filename.endswith(".json.gz"):
        with gzip.open(filename) as handle:
            data = json.load(handle)
    else:
        raise RuntimeError(f"unsupported file {filename}")
    logging.debug(f"Loaded data from {filename} ({len(data)} items)")
    logging.debug(f"Setting relative path with folder {folder}, kwargs={kwargs}")
    join_relative_path(data, folder, **kwargs)
    return data


def save_json(data, filename, folder=None, **kwargs):
    """ Save data to filename (adjust paths relative to folder) 
    """
    folder = os.path.dirname(filename) if folder is None else folder
    if folder != "" and not os.path.exists(folder):
        logging.info(f"Creating folder: {folder}")
        os.makedirs(folder)
    # make a deep-copy since data will be modified
    data = copy.deepcopy(data)
    make_relative_path(data, folder, **kwargs)
    if filename.endswith(".json"):
        with open(filename, "w") as handle:
            json.dump(data, handle, indent=2)
    elif filename.endswith(".json.gz"):
        with gzip.open(filename, 'wt', encoding="ascii") as handle:
            json.dump(data, handle, indent=2)
    logging.debug(f"Saved data to {filename} ({len(data)} items)")


