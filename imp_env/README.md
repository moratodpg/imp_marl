# Add your IMP environment

## Interface imp_env.py

An IMP environment is implemented as a class and includes two main methods: (i) reset and (ii) step. This is already provided in the interface class, [imp_env](imp_env.py).

These methods are included by default because the [wrappers](../imp_wrappers) that will integrate your IMP environment with typical MARL ecosystems needs their definition.

Additionally, you can include any other relevant methods in your environment.

To create a new IMP environment `NewEnv`, inherit imp_env:
```
from imp_env.imp_env import ImpEnv 

class NewEnv(ImpEnv):
```

Feel free to follow this [tutorial](new_imp_env_tutorial.ipynb), where the steps to create an environment are explained in a simple exercise.

You can also check the [guidelines](./pomdp_models/generate_models.ipynb) for generating and storing your own transition model.