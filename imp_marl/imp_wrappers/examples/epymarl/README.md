# EPyMARL

Extended PyMARL [EPyMARL](https://github.com/uoe-agents/epymarl) is an extension of [PyMARL](https://github.com/oxwhirl/pymarl), featuring:
- Additional algorithms (IA2C, IPPO, MAPPO and MAA2C)
- Option for no-parameter sharing between agents
- Flexibility with extra implementation details

For more information, check their [blog](https://agents.inf.ed.ac.uk/blog/epymarl/)

You can run IMP-MARL environments on EPyMARL by:

1. [Clone EPyMARL](https://github.com/uoe-agents/epymarl).
2. Install the required EPyMARL modules according to the requirements file: `pip install -r requirements`. 
3. Install imp_marl package: `pip install git+http://github.com/moratodpg/imp_marl.git`.
4. Add your imp_marl env(s) within EPyMARL environment registry in [envs/__init__.py](https://github.com/uoe-agents/epymarl/tree/main/src/envs/__init__.py):
    ```
    from imp_marl.imp_wrappers.epymarl.epymarl_wrap_ma_struct import EPymarlMAStruct
    REGISTRY["struct_marl"] = partial(env_fn, env=EPymarlMAStruct)
    ```
5. Include the config files associated with your env(s) in the EPyMARL [config folder](https://github.com/uoe-agents/epymarl/tree/main/src/config/envs).

   We provide an example of config file in [struct_uc_3.yaml](imp_marl/imp_wrappers/examples/epymarl/config/struct_uc_3.yaml).

   You can use the config files of PyMARL stored [here](https://github.com/moratodpg/imp_marl/tree/main/papers/neurips_23/pymarl/config/envs) **BUT** you need to add a `map_name` to them as in the example.
7. You can now run it!

   To execute `struct_uc_3` with QMIX, run: 
    ```
    python3 src/main.py --config=qmix --env-config=struct_uc_3
    ```
