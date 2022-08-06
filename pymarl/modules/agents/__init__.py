

REGISTRY = {}

from .rnn_agent import RNNAgent
from .rnn_v_agent import RNNVAgent

REGISTRY["rnn"] = RNNAgent
REGISTRY["rnn_v"] = RNNVAgent
