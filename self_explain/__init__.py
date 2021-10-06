__project__ = "self_explain"
from pkg_resources import get_distribution, DistributionNotFound
try: 
  __version__ = get_distribution(__project__).version
except DistributionNotFound:
  __version__ = "0.0.0"

import os 
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from .self_explain import SelfExplainCharacterizer

def download_benepar():
    """ Download benepar data to ~/nltk_data
    """
    import benepar
    benepar.download('benepar_en3')


def set_resource_limit():
    """ Required by self-explain
    """
    import resource
    rlimit_old = resource.getrlimit(resource.RLIMIT_NOFILE)
    rlimit_new = (4096, rlimit_old[1])
    print(f"setting resource.RLIMIT_NOFILE={rlimit_new} (was {rlimit_old})")
    resource.setrlimit(resource.RLIMIT_NOFILE, rlimit_new)

set_resource_limit()
download_benepar()

