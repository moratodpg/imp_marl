REGISTRY = {}

from .basic_controller import BasicMAC
from .is_controller import ISMAC
from .maven_controller import MavenMAC

REGISTRY["basic_mac"] = BasicMAC
REGISTRY["maven_mac"] = MavenMAC
REGISTRY["is_mac"] = ISMAC
