# encoding:utf-8
import numpy as np
import csv,os,random,json,time
import re,ast
from code_interpreter import CodeInterpreter
from object_filter_gpt4 import ObjectFilter
from datetime import datetime
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    RetryError
)  # for exponential backoff
# openai.api_key = "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"
config = {
        'model': 'gpt-4',
        # 'model': 'gpt-3.5-turbo',
        'temperature': 0,
        'top_p': 0.0,
        'max_tokens': 'inf',

        'system_message': "Imagine you are an artificial intelligence assitant with a python interpreter. So when answering questions, you can choose to generate python code (for example, when there is need to do quantitative evaluation). The generated code should always use print() function to print out the result and keep two decimal places for numbers. The code should be written in python, start with '```python\nimport numpy as np\nimport math\n' and end with '```'. Keep your code and comments concise. When answer step by step, stop whenever you feel there is need to generate python code (for example, where there is need to do quantitative evaluation) and wait for the result from the code execution. Make sure your code will print out something(include failure info like 'nothing found'), especially when you use if logic.",
        # 'load_path': '',
        'save_path': 'chats',
        'debug': True
    }

@retry(wait=wait_exponential(multiplier=20, min=1, max=61), stop=stop_after_attempt(4)) #20s,40s,61s
def get_gpt_response(prompt,gpt_config):
    code_interprter=CodeInterpreter(**gpt_config)
    response,_=code_interprter.call_openai_with_code_interpreter(prompt)
    
    count_response=0
    while not "Now the answer is complete" in response:
        if count_response >=10:
            print("Response does not end with 'Now the answer is complete.' !")
            break
        response,_ = code_interprter.call_openai_with_code_interpreter('')
        response=response['content']
        count_response += 1
        print("count_response:",count_response)

    return response, code_interprter

def round_list(lst,length=2):
    # 给lst的每个元素保留length位小数，返回list类型的变量
    for idx,num in enumerate(lst):
        lst[idx]=round(num,length)
    return list(lst)

def get_observer_position():
    return [0,0,0]

def gen_prompt(metadata, obj_desc):
    # use object filter to get related objects
    of=ObjectFilter()
    relevant_ids,_=of.filter_objects_by_description(description=obj_desc,use_npy_file=False,object_info_list=metadata,to_print=False)
    print("relevant object ids:",relevant_ids)
    objects_related = metadata if (relevant_ids is None) else [obj for obj in metadata if obj["id"] in relevant_ids]

    # calculate scene center
    observer_position=get_observer_position()

    # background part of prompt
    prompt="In a scene, there are objects with quant description based on r-h Cartesian coord sys with x-y-z axes, x-y plane=ground, z-axis=up/down. coords format [x, y, z].\n"

    prompt=prompt+"Observer at coord %s to judge obj orientation.\n"%str(observer_position) 

    prompt=prompt+"objs list:\n"
    # 生成prompt中对物体的定量描述部分（遍历所有相关物体）
    for obj in objects_related:
        # 位置信息，保留2位小数
        center_position=obj['ctr']
        center_position=round_list(center_position,2)
        # size信息，保留2位小数
        size=round_list(obj['size'])
        size=round_list(size,2)

        color=obj['rgb']
        line="%s,id=%s,ctr=%s,size=%s,RGB=%s\n" %(obj['cls'],obj['id'], str(center_position),str(size), str(color))
            
        prompt=prompt+line

    # Instruction part in prompt
    line="Instruction:find the one described object in description: \n\"%s\"\n" %obj_desc
    prompt=prompt+line

    prompt=prompt+"Tips: while multiple objs may appear within the description, it points to only 1 focal object, with the other objects serving to aid in locating or contextualizing it. For instance, spatial relation with other objects might be employed to establish the position or orientation of this focal object."

    # 7 constrains
    prompt=prompt+"\nConsider different constraints in order (1 to 7) & priority (1 highest, 7 lowest):"
    prompt=prompt+"\n1: Obj name(category). Names in description & obj list may differ (e.g. similar names such as 'table' and 'desk', 'trash can' and 'recycling bin', 'coffee table' and 'end table'), so use common sense to find all possible candidate objects, ensure no missing, don't write code. If only 1 object in list has the same/similar category with the one described object, answer it directly, discard other constraints. For instance, with description 'the black bag left to the couch' and only 1 bag in the scene, answer it directly, discard 'black' and 'left' constrains."
    prompt=prompt+"\n2: Horizontal relation like 'next to''farthest''closest''nearest''between''in the middle''at the center'(if given)(not include 'behind''in front of'). Consider only center x,y,z coords of objs, disregard sizes."
    prompt=prompt+"\n3: Color(if given). Be lenient with color, RGB values in obj list & standard RGB value of obj in description may differ significantly. You can use distance in RGB color space as a metric."
    prompt=prompt+"\n4: Size & shape(if given). Be cautious not to make overly absolute judgments about obj size. E.g., 'a tiny trash can' doesn't necessarily refer to smallest one in terms of volume."
    prompt=prompt+"\n5: Direction relation 'left''right'(if given). To judge A on 'left' or 'right' of B, calc vec observer-A & observer-B(both projected to x-y plane). If cross product of vec observer-A & vector observer-B(in this order) has positive z, A on right of B. If z neg, A on left of B. Note that order of cross product matters, put vec observer-A at first. Consider which two objs' left-right relation needs to be determined in sentence, that is, which is A & which is B. DON'T determine left & right relation by compare x or y coords." 
    prompt=prompt+"\n6: Direction relation 'in front of' and 'behind'(if given). Use 'spatially closer' to replace them. To determine which object, P1 or P2, is behind Q, calculate their distances from Q. The one with the smaller distance is behind Q. It is the same for judging 'in front of': also smaller distance. DON'T determine front & behind relation by compare x or y coords." 
    prompt=prompt+"\n7: Vertical relation like 'above'and'under''on''sits on'(if given). Consider only center coords of objs, disregard sizes. Be more lenient with this."
    prompt=prompt+"\nExplicitly go through these 7 constraints. For every constraint, if it is not mentioned in description, tell me and skip; if mentioned, apply this constraint and record the results of each candidates. For constraint 1, use common sense, no code. For others, write code, which should print the metrics of each candidate objects, instead of only print the most possible object id. After going through all constriants, evaluate all results comprehensively basing on 1-7 priority, and choose the unique target object."

    # only one answer
    prompt=prompt+"\nThere is exactly one answer, so if you receive multiple answers, consider other constraints; if get no answers, loosen constraints."
    prompt=prompt+"\nWork this out step by step to ensure right answer."

    # response format
    prompt=prompt+"\nIf the answer is complete, add \"Now the answer is complete -- {'ID':id}\" to the end of your answer, where id is the id of the referred obj."

    print("--------------------------------------------")
    print("Generated prompt:\n"+prompt)
    print("--------------------------------------------")

    return prompt,relevant_ids

def get_obj_id(metadata, obj_desc):
    """given the environment metadata and object description,   return the object id

    Args:
        metadata (list): a list of dictionaries, each   dictionary contains the keys 'id', 'cls', 'ctr', 'size', 'rgb'
        obj_desc (str): a referential text describing the   object to retrieve
    Returns:
        int: the object id
    """
    prompt,relevant_ids=gen_prompt(metadata,obj_desc)
    get_gpt_response_success=True
    try:
        response,code_interpreter=get_gpt_response(prompt,config)
    except RetryError as r:
        print("RetryError!!!")
        print(r)
        last_line="Nonesense"
        get_gpt_response_success=False

    if get_gpt_response_success:
        # print("--------------------------------------------")
        # print("DIALOGUE:")
        # code_interpreter.print_pretext()
        # print("--------------------------------------------")
        last_line = response.splitlines()[-1] if len(response) > 0 else ''
        # print(type(last_line))
        # print("last_line:",last_line)

    last_line_split=last_line.split('--')

    # 使用正则表达式从字符串中提取字典部分
    pattern = r"\{[^\}]*\}"
    match = re.search(pattern, last_line_split[-1])
    if match:
        # 获取匹配的字典字符串
        matched_dict_str = match.group()
        try:
            # 解析字典字符串为字典对象
            extracted_dict = ast.literal_eval(matched_dict_str)
            print(extracted_dict)
            answer_id=extracted_dict['ID']
            print("answer id by gpt:",answer_id)
            # 如果确实以 Now the answer is complete -- {'ID': xxx} 的格式回复了，但是xxx不是数字（例如是None），也只能随机选。
            if not isinstance(answer_id,int):
                if isinstance(answer_id,list) and all([isinstance(e,int) for e in answer_id]):
                    print("Wrong answer format!! random choice.")
                    answer_id=random.choice(answer_id)
                else:
                    print("Wrong answer format!! random choice.")
                    answer_id=random.choice(relevant_ids) 
        except:
            print("Wrong answer format!! No dict found.")
            answer_id=random.choice(relevant_ids)
    else:
        print("Wrong answer format!! No dict found.")
        answer_id=random.choice(relevant_ids)

    print("answer_id returned:",answer_id)
    return answer_id

if __name__=="__main__":
    metadata=[
        {'cls':'table','id':1,'ctr':[1,1,1],'size':[0.1,0.1,0.1], 'rgb':[20,0,255]},
        {'cls':'table','id':2,'ctr':[2,2,2],'size':[0.1,0.1,0.1], 'rgb':[21,0,255]},
        {'cls':'table','id':3,'ctr':[4,4,4],'size':[0.1,0.1,0.1], 'rgb':[200,0,25]},
        {'cls':'chair','id':4,'ctr':[3,3,3],'size':[0.1,0.1,0.1], 'rgb':[0,0,255]},
        {'cls':'bag','id':5,'ctr':[4,3,2],'size':[0.1,0.1,0.1],'rgb':[0,0,255]},
        {'cls':'fridge','id':6,'ctr':[0,3,6],'size':[0.1,0.1,0.1],'rgb':[255,0,255]},
        ]
    description="Find the blue table next to the chair. "
    # 3 tables, table 2 and 3 have same distance to chair(closest),
    # table 1 and 2 are blue, 3 is red
    # correct answer: 2

    get_obj_id(metadata,description)
    