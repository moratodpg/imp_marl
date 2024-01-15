from .coma_learner import COMALearner
from .comaIS_learner import COMAISLearner
from .ddmac_learner import DDMACLearner
from .dmaq_qatten_learner import DMAQ_qattenLearner
from .facmac_learner_discrete import FACMACDiscreteLearner
from .maxqv_learner import MaxQVLearner
from .q_learner import QLearner
from .qv_learner import QVLearner

REGISTRY = {}

REGISTRY["q_learner"] = QLearner
REGISTRY["qv_learner"] = QVLearner
REGISTRY["maxqv_learner"] = MaxQVLearner
REGISTRY["coma_learner"] = COMALearner
REGISTRY["comaIS_learner"] = COMAISLearner
REGISTRY["ddmac_learner"] = DDMACLearner
REGISTRY["dmaq_qatten_learner"] = DMAQ_qattenLearner
REGISTRY["facmac_learner_discrete"] = FACMACDiscreteLearner
