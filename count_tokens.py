import tiktoken
import json
import sys
import os
def num_tokens_from_string(string: str, encoding_name: str=None) -> int:
    """Returns the number of tokens in a text string."""
    # encoding = tiktoken.get_encoding(encoding_name)
    encoding = tiktoken.encoding_for_model("gpt-4")
    num_tokens = len(encoding.encode(string))
    return num_tokens

def count_token_in_dialogue(json_path):
    with open(json_path) as f:
        dia=json.load(f)
    token_input=0
    token_output=0
    for d in dia:
        if d['role'] in ['system','user']:
            token_input+=num_tokens_from_string(d['content'])
        else:
            token_output+=num_tokens_from_string(d['content'])
    return token_input,token_output

def count_token_in_folder(folder_path):
    file_names=os.listdir(folder_path)
    file_count=0
    token_input_total=0
    token_output_total=0
    for file in file_names:
        if 'refer' in file:
            token_input,token_output=count_token_in_dialogue(os.path.join(folder_path,file))
            token_input_total+=token_input
            token_output_total+=token_output
            file_count+=1

    return token_input_total,token_output_total,file_count

folder_path="/share/data/ripl/vincenttann/sr3d/eval_results_sr3d_4_p_testset/2023-09-16-00-37-29/2023-09-16-00-37-29_dialogue_jsons"
token_input_total,token_output_total,file_count=count_token_in_folder(folder_path)
print(token_input_total,token_output_total,file_count)