import json
import openai
import glob
import os
import pandas as pd

# prepare file list
results_folder = 'eval_results_nr3d_4_p_trainset'
timestamps = [
    # '2023-10-29-22-36-47',
    '2023-11-14-15-40-01',
    '2023-11-14-23-15-07',
    '2023-11-14-23-15-16',
    '2023-11-14-23-15-26',
    '2023-11-14-23-15-36',
    '2023-11-14-23-15-47',
    '2023-11-15-01-11-11',
]

all_refer_success_list = []
all_refer_correction_list = []

for timestamp in timestamps:
    
    print(timestamp)

    dialugue_folder = os.path.join(results_folder, timestamp, '{}_dialogue_jsons'.format(timestamp))
    refer_success_list = glob.glob(dialugue_folder + '/*refer_success.json')
    refer_correction_list = glob.glob(dialugue_folder + '/*refer_correction.json')
    
    all_refer_success_list += refer_success_list
    all_refer_correction_list += refer_correction_list

    # print(refer_success_list)
    # print(refer_correction_list)
    # print(len(refer_success_list))
    # print(len(refer_correction_list))

print('length all_refer_success_list: ', len(all_refer_success_list))
print('length all_refer_correction_list: ', len(all_refer_correction_list))

# read in as message
def read_json_file_as_message(filename):
    with open(filename, 'r') as f:
        data = json.load(f)
    return {'messages': data}

# for example in all_refer_success_list[:1]:
#     print(example)
#     print(read_json_file_as_message(example))

# filter data for different settings:
# 1. success examples with tips
# 2. success + correct examples with tips
# 3. success examples without tips
# 4. success + correct examples without tips

# setting 1
all_refer_success_messages = [read_json_file_as_message(refer_success_example) for refer_success_example in all_refer_success_list]
print('length all_refer_success_messages: ', len(all_refer_success_messages))

# for self-correct examples with tips, just need to remove the one line that asks it to re-generate
def remove_correction_prefix(data_json):
    data_json['messages'].pop(2)
    return data_json

all_refer_correction_messages = []
for refer_correction_example in all_refer_correction_list:
    message = read_json_file_as_message(refer_correction_example)
    message = remove_correction_prefix(message)
    all_refer_correction_messages.append(message)
print('length all_refer_correction_messages: ', len(all_refer_correction_messages))
# print(all_refer_correction_messages[0]['messages'][1])
# print(all_refer_correction_messages[0]['messages'][2])
# print(all_refer_correction_messages[0]['messages'][3])

# setting 2
all_refer_success_correction_messages = all_refer_success_messages + all_refer_correction_messages
print('length all_refer_success_correction_messages: ', len(all_refer_success_correction_messages))

# settings 3,4
def filter_out_rules(json_data):
    user_input = json_data['messages'][0]['content'] # get user input
    # start_phrase = 'Tips'
    start_phrase = 'Below are some tips to help you reasoning and finding the object'
    # end_phrase = 'basing on 1-7 priority, and choose the unique target object.\n'
    end_phrase = 'Wall front=side of plane where obj exist.'
    user_input_without_rules = user_input.split(start_phrase)[0] + user_input.split(end_phrase)[-1]
    json_data['messages'][0]['content'] = user_input_without_rules

    return json_data

# example_data = read_json_file_as_message(all_refer_correction_list[0])
# print(filter_out_rules(example_data)['messages'][0]['content'])

# setting 3
no_rules_all_refer_success_messages = [filter_out_rules(read_json_file_as_message(refer_success_example)) for refer_success_example in all_refer_success_list]
no_rules_all_refer_correction_messages = [filter_out_rules(remove_correction_prefix(read_json_file_as_message(refer_correction_example))) for refer_correction_example in all_refer_correction_list]
print('length no_rules_all_refer_success_messages: ', len(no_rules_all_refer_success_messages))
print('length no_rules_all_refer_correction_messages: ', len(no_rules_all_refer_correction_messages))

# setting 4
no_rules_all_refer_success_correction_messages = no_rules_all_refer_success_messages + no_rules_all_refer_correction_messages
print('length no_rules_all_refer_success_correction_messages: ', len(no_rules_all_refer_success_correction_messages))

# write to jsonl
def write_jsonl(data_list: list, filename: str) -> None:
    with open(filename, "w") as out:
        for ddict in data_list:
            jout = json.dumps(ddict) + "\n"
            out.write(jout)

# setting 1
all_refer_success_train_fn = "finetune_files/all_refer_success_train.jsonl"
write_jsonl(all_refer_success_messages[:380], all_refer_success_train_fn)
all_refer_success_val_fn = "finetune_files/all_refer_success_val.jsonl"
write_jsonl(all_refer_success_messages[380:], all_refer_success_val_fn)

# setting 2
all_refer_success_correction_train_fn = "finetune_files/all_refer_success_correction_train.jsonl"
write_jsonl(all_refer_success_messages[:500], all_refer_success_correction_train_fn)
all_refer_success_correction_val_fn = "finetune_files/all_refer_success_correction_val.jsonl"
write_jsonl(all_refer_success_messages[500:], all_refer_success_correction_val_fn)

# setting 3
no_rules_all_refer_success_train_fn = "finetune_files/no_rules_all_refer_success_train.jsonl"
write_jsonl(no_rules_all_refer_success_messages[:380], no_rules_all_refer_success_train_fn)
no_rules_all_refer_success_val_fn = "finetune_files/no_rules_all_refer_success_val.jsonl"
write_jsonl(no_rules_all_refer_success_messages[380:], no_rules_all_refer_success_val_fn)

# setting 4
no_rules_all_refer_success_correction_train_fn = "finetune_files/no_rules_all_refer_success_correction_train.jsonl"
write_jsonl(no_rules_all_refer_success_correction_messages[:500], no_rules_all_refer_success_correction_train_fn)
no_rules_all_refer_success_correction_val_fn = "finetune_files/no_rules_all_refer_success_correction_val.jsonl"
write_jsonl(no_rules_all_refer_success_correction_messages[500:], no_rules_all_refer_success_correction_val_fn)