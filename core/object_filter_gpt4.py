import numpy as np
import os,json
import re
import logging
from gpt_dialogue import Dialogue
import openai
from tenacity import (
    retry,
    before_sleep_log,
    stop_after_attempt,
    wait_random_exponential,
    wait_exponential,
    wait_exponential_jitter,
    RetryError
)  # for exponential backoff

openai.api_key = os.getenv("OPENAI_API_KEY")

logger = logging.getLogger(__name__+'logger')
logger.setLevel(logging.ERROR)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.ERROR)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

class ObjectFilter(Dialogue):
    def __init__(self, model='gpt-4'):
        config = {
        # 'model': 'gpt-4',
        # 'model': 'gpt-4-1106-preview',
        'model': model,
        'temperature': 0,
        'top_p': 0.0,
        'max_tokens': 8192,
        # 'load_path': './object_filter_pretext.json',
        'load_path': './object_filter_pretext_new.json',
        'debug': False
        }
        super().__init__(**config)
    
    def extract_all_int_lists_from_text(self,text) ->list:
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

    def extract_dict_from_text(self,text) ->dict:
        # Use regular expression to match the dictionary in the text
        match = re.search(r'{\s*(.*?)\s*}', text)
        if match:
            # Get the matched dictionary content
            dict_str = match.group(1)
            # Convert the dictionary string to an actual dictionary object
            try:
                result_dict = eval('{' + dict_str + '}')
                return result_dict
            except Exception as e:
                print(f"Error converting string to dictionary: {e}")
                return None
        else:
            print("No dictionary found in the given text.")
            return None

    @retry(wait=wait_exponential_jitter(initial=20, max=120, jitter=20), stop=stop_after_attempt(5), before_sleep=before_sleep_log(logger,logging.ERROR)) #20s,40s,80s,120s + random.uniform(0,20)
    def filter_objects_by_description(self,description,use_npy_file,objects_info_path=None,object_info_list=None,to_print=True):
        # first, create the prompt
        print("looking for relevant objects based on description:\n'%s'"%description)
        prompt=""
        prompt=prompt+"description:\n'%s'\nobject list:\n"%description
        # load object info data and add to prompt
        if use_npy_file:
            data=np.load(objects_info_path,allow_pickle=True)
            for obj in data:
                if obj['label']=='object':
                    continue
                line="name=%s,id=%d; "%(obj['label'],obj['id'])
                prompt=prompt+line
        else: # object info list given, used for robot demo
            data=object_info_list
            for obj in data:
                label=obj.get('cls')
                if label is None:
                    label=obj.get('label')
                # if obj['cls']=='object':
                #     continue
                if label in ['object','otherfurniture','other','others']:
                    continue
                line="name=%s,id=%d; "%(label,obj['id'])
                prompt=prompt+line
        
        
        # get response from gpt
        response,token_usage=self.call_openai(prompt)
        response=response['content']
        # print("response:",response)
        last_line = response.splitlines()[-1] if len(response) > 0 else ''

        # exract answer(list/dict) from the last line of response
        # answer=self.extract_all_int_lists_from_text(last_line)
        answer=self.extract_dict_from_text(last_line)
        if to_print:
            self.print_pretext()
            print("answer:",answer)
            print("\n\n")
        if len(answer)==0:
            answer=None
        return answer,token_usage
    

    
if __name__ == "__main__":
    # scanrefer_path="/share/data/ripl/vincenttann/sr3d/data/scanrefer/ScanRefer_filtered_sampled50.json"
    scanrefer_path="/share/data/ripl/vincenttann/sr3d/data/scanrefer/ScanRefer_filtered_train_sampled1000.json"
    with open(scanrefer_path, 'r') as json_file:
        scanrefer_data=json.load(json_file)
    
    from datetime import datetime
    # 记录时间作为文件名
    current_time = datetime.now()
    formatted_time = current_time.strftime("%Y-%m-%d-%H-%M-%S")
    print("formatted_time:",formatted_time)
    folder_path="/share/data/ripl/vincenttann/sr3d/object_filter_dialogue/%s/"%formatted_time
    os.makedirs(folder_path)

    for idx,data in enumerate(scanrefer_data):
        print("processing %d/%d..."%(idx+1,len(scanrefer_data)))
        description=data['description']
        scan_id=data['scene_id']
        target_id=data['object_id']
        # path="/share/data/ripl/scannet_raw/train/objects_info_gf/objects_info_gf_%s.npy"%scan_id
        path="/share/data/ripl/scannet_raw/train/objects_info/objects_info_%s.npy"%scan_id
        of=ObjectFilter()
        of.filter_objects_by_description(path,description)
        object_filter_json_name="%d_%s_%s_object_filter.json"%(idx,scan_id,target_id)
        of.save_pretext(folder_path,object_filter_json_name)