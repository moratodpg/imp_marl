import glob
import json
import os


def compare_dicts(dict_of_dicts):
    # Get the keys present in all of the dictionaries
    all_keys = set()
    for inner_dict in dict_of_dicts.values():
        all_keys |= set(inner_dict.keys())
    all_keys.remove("unique_token")
    all_keys.remove("seed")
    result = {}
    for key in all_keys:
        values = [
            str(inner_dict[key])
            for inner_dict in dict_of_dicts.values()
            if key in inner_dict
        ]
        if len(set(values)) > 1:
            result[key] = {}
            for name, inner_dict in dict_of_dicts.items():
                if key in inner_dict:
                    result[key][name] = inner_dict[key]
                else:
                    result[key][name] = None
    return result


sacred_dir = "results/sacred"
exp = "dqn_sarl_struct_sarl_uc_5"
config_json_files = glob.glob(
    os.path.join(sacred_dir, "*" + exp + "*/*config.json"), recursive=True
)
print(len(config_json_files))

expe = {}
for i in config_json_files:
    try:
        with open(i, "r") as file:
            config = json.load(file)
            expe[i] = config
    except Exception as e:
        print(e)

result = compare_dicts(expe)
# Transpose the table
transposed_result = {}
for key, value in result.items():
    for name, inner_value in value.items():
        if name not in transposed_result:
            transposed_result[name] = {}
        transposed_result[name][key] = inner_value

transposed_result = dict(sorted(transposed_result.items()))
# transposed_result = dict(sorted(transposed_result.items(), key=lambda item: (item[1]['n_head'],item[1]['adv_hypernet_layers'],item[1]['mixing_embed_dim'])))# Print the transposed table
# agent_fc1           	epsilon_anneal_time 	use_cuda            	rnn_hidden_dim      	buffer_size         	agent_fc2
max_key_length = max([len(key) for key in result])
ljust = max_key_length + 10
print()
print("names".ljust(len(list(transposed_result.keys())[0]) - 15 - 12 - 6), "\t", end="")
print(
    "max return".ljust(len(list(transposed_result.keys())[0]) - 15 - 12 - 6),
    "\t",
    end="",
)
for key in result:
    print(key.ljust(ljust), "\t", end="")
print()

for idx, (name, value) in enumerate(transposed_result.items()):
    print(name[15:-12].ljust(ljust), "\t", end="")
    try:
        with open(
            os.path.join(os.path.join(os.path.dirname(name), "info.json")), "r"
        ) as file:
            info = json.load(file)
            print(
                str(max([d["value"] for d in info["test_return_mean"]])).ljust(ljust),
                "\t",
                end="",
            )
    except Exception as e:
        print(e)
    for key in result:
        if key in value:
            print(str(value[key]).ljust(ljust), "\t", end="")
        else:
            print("\t", end="")
    print()
