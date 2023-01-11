REGISTRY = {}

from .basic_controller import BasicMAC
from .facmac_controller import FacmacMAC
from .is_controller import ISMAC

REGISTRY["basic_mac"] = BasicMAC
REGISTRY["facmac_mac"] = FacmacMAC
REGISTRY["is_mac"] = ISMAC
