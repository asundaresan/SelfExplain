__project__ = "self_explain"
from pkg_resources import get_distribution, DistributionNotFound
try: 
  __version__ = get_distribution(__project__).version
except DistributionNotFound:
  __version__ = "0.0.0"

from .self_explain import SelfExplainCharacterizer

