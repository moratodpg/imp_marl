import os
import glob
import sys
import json


# def compare_dicts(dict_of_dicts):
#     # Get the keys present in all of the dictionaries
#     all_keys = set()
#     for inner_dict in dict_of_dicts.values():
#         all_keys |= set(inner_dict.keys())
#     # Print out a table showing the values for each key and dictionary, but only for keys where the values are different between the dictionaries
#     # Determine the maximum key length and value length for each dictionary
#     selected_dicts = {name: {key: inner_dict[key] for key in inner_dict if key in all_keys} for name, inner_dict in
#                       dict_of_dicts.items()}
#     max_key_lengths = {}
#     max_value_lengths = {}
#     for name, inner_dict in selected_dicts.items():
#         max_key_lengths[name] = max([len(key) for key in inner_dict])
#         max_value_lengths[name] = max([len(str(value)) for value in inner_dict.values()])
#     # Determine the maximum key length and value length across all dictionaries
#     max_key_length = max(max_key_lengths.values())
#     max_value_length = max(max_value_lengths.values())
#     print(max_key_length)
#     print(max_value_length)
#     # Print the table header
#     print("Key", end='')
#     for name in dict_of_dicts:
#         print(" " * (max_key_length - len(name) + 1), name, end='')
#     print()
#     # Print the rows of the table
#     for key in all_keys:
#         values = [str(inner_dict[key]) for inner_dict in dict_of_dicts.values() if key in inner_dict]
#         if len(set(values)) > 1:
#             print(key, end='')
#             for name, inner_dict in dict_of_dicts.items():
#                 if key in inner_dict:
#                     value = inner_dict[key]
#                     print(" " * (max_value_length - len(str(value)) + 1), value, end='')
#                 else:
#                     print(" " * (max_value_length + 1), end='')
#             print()

def compare_dicts(dict_of_dicts):
    # Get the keys present in all of the dictionaries
    all_keys = set()
    for inner_dict in dict_of_dicts.values():
        all_keys |= set(inner_dict.keys())
    all_keys.remove("unique_token")
    all_keys.remove("seed")
    result = {}
    for key in all_keys:
        values = [str(inner_dict[key]) for inner_dict in dict_of_dicts.values() if key in inner_dict]
        if len(set(values)) > 1:
            result[key] = {}
            for name, inner_dict in dict_of_dicts.items():
                if key in inner_dict:
                    result[key][name] = inner_dict[key]
                else:
                    result[key][name] = None
    return result


sacred_dir = "results/sacred"
exp = "iql_uc_10_struct_uc_10"
config_json_files = glob.glob(
    os.path.join(sacred_dir, "*" + exp + "*/*config.json"), recursive=True)
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

transposed_result = dict(sorted(transposed_result.items(), key=lambda item: (item[1]['epsilon_anneal_time'], item[1]['agent_fc1'], item[1]['rnn_hidden_dim'], item[1]['agent_fc2'])))# Print the transposed table
max_key_length = max([len(key) for key in result])
ljust = max_key_length
print("names".ljust(len(list(transposed_result.keys())[0])-15-12), "\t", end='')
for key in result:
    print(key.ljust(ljust), "\t", end='')
print()
for idx, (name, value) in enumerate(transposed_result.items()):
    print(name[15:-12].ljust(ljust), "\t", end='')
    for key in result:
        if key in value:
            print(str(value[key]).ljust(ljust), "\t", end='')
        else:
            print("\t", end='')
    print()
