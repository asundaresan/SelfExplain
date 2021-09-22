""" JSON utilities 
""" 
import json 
import gzip 
import bz2
import logging 
import os
import numpy as np
import copy

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

  

def join_relative_path(data, folder, keys=[ ], suffixes=[ "filename", "filepattern", "filepatterns" ], prefixes=[ ]):
  """ Iterate recursively through data and whenever there is a key match 
  (i.e. key in keys or key.endswith(suffix) or key.startswith(prefix))
  and type(value) is str, join the path to folder to get full path.
  """
  if isinstance(data, dict):
    for k, v in data.items():
      if key_match(k, keys=keys, suffixes=suffixes, prefixes=prefixes):
        logging.log(1, "matched %s: %s" % (k, v))
        if isinstance(v, str): 
          data[k] = os.path.join(folder, v)
        elif isinstance(v, list): 
          data[k] = list(os.path.join(folder, v2) for v2 in v)
      join_relative_path(v, folder, keys=keys)
  elif isinstance(data, list):
    for d in data:
      join_relative_path(d, folder, keys=keys)


def make_relative_path(data, folder, keys=[ ], suffixes=[ "filename" ], prefixes=[ ] ):
  """ Iterate recursively through data and whenever a (key="filename", value=str)
      pair is encountered, compute path relative to folder. 
  """
  if isinstance(data, dict):
    for k, v in data.items():
      if isinstance(v, str) and key_match(k, keys=keys, suffixes=suffixes, prefixes=prefixes):
        data[k] = os.path.relpath(v, folder)
        logging.debug("setting %s: %s -> %s" % (k, v, data[k]))
      make_relative_path(v, folder, keys=keys)
  elif isinstance(data, list):
    for d in data:
      make_relative_path(d, folder, keys=keys)



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
    with gzip.open(filename, "rt") as handle:
      data = json.load(handle)
  elif filename.endswith(".json.bz2"):
    with bz2.open(filename, "rt") as handle:
      data = json.load(handle)
  else: 
      raise RuntimeError(f"Filename type not supported: {filename}")
  logging.debug("Loaded data from %s (%d items)" % (filename, len(data)))
  logging.debug("Setting relative path with folder %s, kwargs=%s" % (folder, kwargs))
  join_relative_path(data, folder, **kwargs)
  return data


def save_json(data, filename, keys=[ ], suffixes=[ ], prefixes=[ "file" ], folder=None):
  """ Save data to filename (adjust paths relative to folder) 
  """
  folder = os.path.dirname(filename) if folder is None else folder
  if folder != "" and not os.path.exists(folder):
    logging.info("Creating folder: %s" % folder)
    os.makedirs(folder)
  make_relative_path(data, folder, keys=keys, suffixes=suffixes, prefixes=prefixes)
  if filename.endswith(".json"):
    with open(filename, "w") as handle:
      json.dump(data, handle, indent=2)
  elif filename.endswith(".json.gz"):
    with gzip.open(filename, 'wt', encoding="ascii") as handle:
      json.dump(data, handle, indent=2)
  logging.debug("Saved data to %s (%d items)" % (filename, len(data)))


