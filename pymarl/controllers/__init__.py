REGISTRY = {}

from .basic_controller import BasicMAC
from .is_controller import ISMAC

REGISTRY["basic_mac"] = BasicMAC
REGISTRY["is_mac"] = ISMAC
