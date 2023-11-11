import os
import re
import glob
import json
import numpy as np

def extract_all_int_lists_from_text(text) ->list:
    # 匹配方括号内的内容
    pattern = r'\[([^\[\]]+)\]'
    matches = re.findall(pattern, text)

    int_lists = []

    for match in matches:
        elements = match.split(',')
        int_list = []

        for element in elements:
            element = element.strip()
            try:
                int_value = int(element)
                int_list.append(int_value)
            except ValueError:
                pass
            
        if len(int_list) == len(elements):
            int_lists = int_lists + int_list

    return int_lists

object_filter_folder = 'eval_results_nr3d_codellama_7b_instruct_p_testset'
timestamps = ['2023-11-06-22-42-54', '2023-11-06-22-45-07', '2023-11-06-22-46-33', '2023-11-06-22-46-53', '2023-11-06-22-47-19', '2023-11-06-22-47-43', '2023-11-07-00-21-20', '2023-11-07-00-21-39', '2023-11-07-00-22-21', '2023-11-07-00-22-23', '2023-11-07-00-22-43', '2023-11-07-00-23-20', '2023-11-07-00-24-09', '2023-11-07-00-24-11', '2023-11-07-00-25-20', '2023-11-07-00-26-58', '2023-11-07-00-27-16', '2023-11-07-00-27-27', '2023-11-07-00-27-29', '2023-11-07-00-27-36']

before_filter_object_list_length_list = []
after_filter_object_list_length_list = []

for timestamp in timestamps:
    dialogue_dir = os.path.join(object_filter_folder, timestamp, '{}_dialogue_jsons'.format(timestamp))
    object_filter_jsons = glob.glob("{}/*_object_filter.json".format(dialogue_dir))
    # print(object_filter_jsons)

    for object_filter_json in object_filter_jsons:
        # print(object_filter_json)
        dialogue_list = json.load(open(object_filter_json))
        input_dialogue = dialogue_list[1]
        # print(input_dialogue['content'])
        num_objects = input_dialogue['content'].count("id=")
        # print(num_objects)
        before_filter_object_list_length_list.append(num_objects)

        last_dialogue = dialogue_list[-1]
        # print(last_dialogue['content'])
        int_list = extract_all_int_lists_from_text(last_dialogue['content'])
        # print(int_list)
        after_filter_object_list_length_list.append(len(int_list))
        


print(before_filter_object_list_length_list)
print(after_filter_object_list_length_list)
before_filter_object_list_length_np = np.array(before_filter_object_list_length_list)
print('mean objects before filtered: ',before_filter_object_list_length_np.mean())
print('mean before filer chance: ', (1.0/before_filter_object_list_length_np).mean())
after_filter_object_list_length_np = np.array(after_filter_object_list_length_list)
print('mean objects filtered: ',after_filter_object_list_length_np.mean())
print('mean after filer chance: ', (1.0/after_filter_object_list_length_np).mean())
