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
# openai.api_key = "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"

logger = logging.getLogger(__name__+'logger')
logger.setLevel(logging.ERROR)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.ERROR)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

class ObjectFilter(Dialogue):
    def __init__(self):
        config = {
        'model': 'gpt-4',
        'temperature': 0,
        'top_p': 0.0,
        'max_tokens': 'inf',
        'load_path': './object_filter_pretext.json',
        'system_message': "You'll receive two inputs: a description & a list of objects. Your job is to find all relevant objects from this list according to a description.\nA defination of 'relevant' - obj1 and obj2 are relevant if one of the following is true: \n1.obj1 and obj2 have exactly the same name; \n2.obj1 and obj2 are synonyms(e.g. 'desk' and 'table', 'trashcan' and 'recycling bin', 'couch' and 'sofa'); \n3.obj1 is a abbreviation of obj2 (e.g. 'fridge' and 'refrigerator', 'tv' and 'television'); \n4. obj1 and obj2 are closely related conceptual categories (e.g. 'cabinet' and 'kitchen cabinet', 'coffee table' and 'end table', 'sofa chair' and 'chair').\nSo desk and table are relevant, trash can and recycling bin are relevant, couch and sofa are relevant, fridge and refridgerator are relevant, cabinet and kitchen cabinet are relevant, coffee table and  end table are relevant, sofa chair, chair and sofa are relevant.\nYou should follow these steps:\n1. Identify all objects in description. Note that objects could include 'man''wall', but not include 'viewer''observer'.\n2. Go through the object list to check each object, if it is relevant to any object in description, pick it out, and keep going until you finish the whole list. You should output 'name id:relevant'/'name id:not' for each object.\n3. For each object in description, output the ids of objects relevant to it. If no relevant object of it found, double check the list and find some that are most possibly relevant, and pick them out.\n4. Summarize the ids of all the ids (include possibly relevant) in step 3 and put them into one list, then add 'Here is the list of relevant objects ids -- [id1,id2,...idn]' to the end of your answer(in a new line, strictly follow the format)\nYou should tell me the result of each step above. " ,
        'save_path': 'chats',
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

    @retry(wait=wait_exponential_jitter(initial=20, max=120, jitter=20), stop=stop_after_attempt(5), before_sleep=before_sleep_log(logger,logging.ERROR)) #20s,40s,80s,120s + random.uniform(0,20)
    def filter_objects_by_description(self,description,use_npy_file,objects_info_path=None,object_info_list=None,to_print=True):
        print("looking for relevant objects based on description:\n%s"%description)
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
        
        
        # print("prompt: ",prompt)
        # response=self.get_gpt_response(prompt)
        response,token_usage=self.call_openai(prompt)
        response=response['content']
        # print("response:",response)
        last_line = response.splitlines()[-1] if len(response) > 0 else ''

        # last_line_split=last_line.split('--')[-1]

        # # 找到方括号内的列表部分
        # start_index = last_line_split.find('[')
        # end_index = last_line_split.find(']')
        # list_str = last_line_split[start_index:end_index + 1]

        # # 将字符串转换为列表
        # list_elements = eval(list_str)

        # # 检查列表中的元素是否都是整数
        # all_integers = all(isinstance(elem, int) for elem in list_elements)

        list_elements=self.extract_all_int_lists_from_text(last_line)
        if to_print:
            self.print_pretext()
            print("answer:",list_elements)
            print("\n\n")
        answer=list_elements if len(list_elements)!=0 else None
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