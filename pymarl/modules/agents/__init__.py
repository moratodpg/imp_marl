
REGISTRY = {}

from .rnn_agent import RNNAgent
from .rnn_v_agent import RNNVAgent
from .rnn_agent_sarl import RNNAgentSARL

REGISTRY["rnn"] = RNNAgent
REGISTRY["rnn_v"] = RNNVAgent
REGISTRY["rnn_sarl"] = RNNAgentSARL
