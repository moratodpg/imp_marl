# EPyMARL

Extended PyMARL [EPyMARL](https://github.com/uoe-agents/epymarl) is an extension of [PyMARL](https://github.com/oxwhirl/pymarl), featuring:
- Additional algorithms (IA2C, IPPO, MAPPO and MAA2C)
- Option for no-parameter sharing between agents
- Flexibility with extra implementation details

For more information, check their [blog](https://agents.inf.ed.ac.uk/blog/epymarl/)

You can run IMP-MARL environments on EPyMARL by:

1. Create a new virtual environment.
2. Install imp_marl package: `pip install git+http://github.com/moratodpg/imp_marl.git`.
3. [Clone EPyMARL](https://github.com/uoe-agents/epymarl).
4. Install the required EPyMARL modules according to the requirements file: `pip install -r requirements`.
5. Register your imp_marl env(s) within EPyMARL. Do that on `/src/envs/__init__.py`:
    ```
    from imp_marl.imp_wrappers.epymarl_wrapper.epymarl_wrap_ma_struct import EPymarlMAStruct
    def env_fn(env, **kwargs) -> MultiAgentEnv:
    return env(**kwargs)
    REGISTRY = {}
    REGISTRY["struct_marl"] = partial(env_fn, env=EPymarlMAStruct)
    ```
6. Include the config files associated with your env(s) in `/src/config/envs`. As reference, you can find here the config file for a struct env with three uncorrelated structural components `struct_uc_3.yaml`. Other config files can be found [here](https://github.com/moratodpg/imp_marl/tree/main/papers/neurips_23/pymarl/config/envs). 
7. You can now run it! To execute for example struct_uc_3 with QMIX, run: 
    ```
    python3 src/main.py --config=qmix --env-config=struct_uc_3
    ```
