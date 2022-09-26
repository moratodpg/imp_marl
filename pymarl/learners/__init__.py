from .maxqv_learner import MaxQVLearner
from .qv_learner import QVLearner
from .q_learner import QLearner
from .coma_learner import COMALearner
from .comaIS_learner import COMAISLearner
from .maven_learner import MavenLearner
REGISTRY = {}

REGISTRY["q_learner"] = QLearner
REGISTRY["qv_learner"] = QVLearner
REGISTRY["maxqv_learner"]= MaxQVLearner
REGISTRY["coma_learner"] = COMALearner
REGISTRY["comaIS_learner"] = COMAISLearner
