REGISTRY = {}

from .basic_controller import BasicMAC
from .maven_controller import MavenMAC

REGISTRY["basic_mac"] = BasicMAC
REGISTRY["maven_mac"] = MavenMAC
