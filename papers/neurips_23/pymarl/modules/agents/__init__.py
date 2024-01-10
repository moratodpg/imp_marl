REGISTRY = {}

from .rnn_agent import RNNAgent
from .rnn_agent_sarl import RNNAgentSARL
from .rnn_v_agent import RNNVAgent

REGISTRY["rnn"] = RNNAgent
REGISTRY["rnn_v"] = RNNVAgent
REGISTRY["rnn_sarl"] = RNNAgentSARL
