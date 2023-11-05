# encoding:utf-8
import numpy as np
import csv
from gpt_dialogue import Dialogue
from datetime import datetime
import openai
import time
import random
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff

SCANNET_DATA_ROOT="/share/data/ripl/scannet_raw/train/"
SCRIPT_ROOT="/share/data/ripl/vincenttann/sr3d/"

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def call_openai(user_prompt, model='gpt-4', temperature=0, top_p=0.1, system_message=''):
    pretext= pretext = [{"role": "system", "content": system_message}]
    user_message = [{"role": "user", "content": user_prompt}]
    completion = openai.ChatCompletion.create(
        model= model,
        messages=pretext + user_message,
        temperature=temperature,
        top_p=top_p,
    )
    assistant_response = completion.choices[0].message
    return assistant_response

def read_csv_with_index(file_path):
    data = {}  # 用字典来保存索引后的数据
    with open(file_path, 'r', newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        headers = next(reader)  # 读取第一行作为header
        for index, row in enumerate(reader, start=1):  # 从第二行开始遍历内容，同时记录行索引
            data[index] = dict(zip(headers, row))  # 使用字典的方式，将header和对应行的内容组合成键值对
    # print(len(data))
    return data

def gen_GPT_prompt_sr3d(sr3d_data_index,scannet_data_root,to_print=True,to_seperate_numbers=False):
    """
    对于sr3d中的制定数据，返回prompt以及其他相关信息
    """
    
    # 读入sr3d数据集
    if to_print:
        print("sr3d_data_index:",sr3d_data_index)
    csv_path=SCRIPT_ROOT+"data/sr3d_train_allocentric.csv"
    data=read_csv_with_index(csv_path)[sr3d_data_index]
    
    # 读入scan_id
    scan_id=data["scan_id"]
    if to_print:
        print("scan_id:",scan_id)

    # 读入refered class and object ids
    target_class=data["instance_type"]
    target_id=data["target_id"]
    distractor_ids=eval(data["distractor_ids"])

    # 读入anchor classes and ids
    anchor_classes=data["anchors_types"]
    anchor_ids=eval(data["anchor_ids"])

    # 读入utterance
    utterance=data["utterance"]

    # 读入reference type
    reference_type=data["coarse_reference_type"]

    
    # 读入事先准备好的物体信息，即npy文件
    npy_path=scannet_data_root+"/objects_info/objects_info_"+scan_id+".npy"
    objects_info=np.load(npy_path,allow_pickle=True)

    # 整合所有物体信息
    objects_related=[]
    objects_related.append(objects_info[int(target_id)])
    for id in distractor_ids:
        objects_related.append(objects_info[int(id)])
    for id in anchor_ids:
        objects_related.append(objects_info[int(id)])
    if to_print:
        print("object_related:\n",objects_related)

    # 生成prompt
    prompt=scan_id + " has objects in it and I'll give you some quantitative descriptions. " +\
    "All quantitative descriptions are based on right-handed Cartesian coordinate system with x-y-z axes, " + \
    "where x represents left-right, y represents forward-backward, and z represents up-down. " + \
    "Objects are:\n"

    # prompt中对物体的定量描述
    if to_seperate_numbers:
        for obj in objects_related:
            quan_info=obj["quan_info"]
            quan_info_seperate=['','','','','','']
            #保留三位小数，并在相邻字符间插入空格
            for idx,num in enumerate(quan_info):
                quan_info_seperate[idx]=' '.join(str(round(num,3))) 
            line="A %s with id %s, its center position is %s, and its size in x,y,z direction is %s.\n" %(obj["label"],obj["id"],quan_info_seperate[0:3],quan_info_seperate[3:] )
            prompt=prompt+line

    else:
        for obj in objects_related:
            quan_info=obj["quan_info"]
            quan_info_round=['','','','','','']
            #保留三位小数，并在相邻字符间插入空格
            for idx,num in enumerate(quan_info):
                quan_info_round[idx]=round(num,2)
            line="A %s with id %s, its center position is %s, and its size in x,y,z direction is %s.\n" %(obj["label"],obj["id"],quan_info_round[0:3],quan_info_round[3:] )
            prompt=prompt+line

    # prompt中的要求
    line="Find the referred object in the following sentence:\n" # and display its id only:\n"
    prompt=prompt+line+utterance+ '.' 
    prompt=prompt+"\nYou should work this out in a step by step way to make sure we have the right answer."#, then display the id in a seperated line. "
    prompt=prompt+"\nYou should calculate the result in each step and tell me the exact final result."
    # prompt=prompt+"But Do Not response anything except the id."
    prompt=prompt+"\nIn the last line of your response, there should Only be a python dictionary in format: {'ID':id}, where id is the id of the referred object."
    
    # prompt=prompt+"\n Do not stop before you find the right id."

    if to_print:
        print("--------------------------------------------")
        print("Generated prompt:\n"+prompt)
        print("--------------------------------------------")
        print("Right answer:",target_id)
    info=(scan_id,target_id,distractor_ids,reference_type,utterance,csv_path)
    return prompt,info


def dialogue_with_GPT(scannet_data_root=SCANNET_DATA_ROOT):
    """
    用对话的方式获取答案并比较
    """

    # 创建dialogue实例
    config = {
        'model': 'gpt-4',
        # 'model': 'gpt-3.5-turbo',
        'temperature': 0,
        'top_p': 0.1,
        'max_tokens': 'inf',
        'system_message': '',
        # 'load_path': 'chats/dialogue_an apple.json',
        'save_path': 'chats',
        'debug': False
    }
    

    # # 告知GPT背景信息
    # background_prompt=\
    # "I wiil describe some scenes and some objects in the scene, and I want you to analyse the spatial relationship of the objects in the scene and answer my questions." + \
    # " All descriptions are in right-handed Cartesian coordinate system with x-y-z axes, " + \
    # "where x represents left-right, y represents forward-backward, and z represents up-down. " + \
    # "In each scene, I will tell you the center position and size in x-y-z direction of the objects."
    # dialogue.call_openai(background_prompt)

    while True:
        # 生成sr3d中指定问题的prompt
        sr3d_line_number=input("Line number in sr3d_train.csv:")
        if sr3d_line_number == 'exit':
            break
        prompt,info=gen_GPT_prompt_sr3d(int(sr3d_line_number)-1, scannet_data_root, to_print=True, to_seperate_numbers=False)

        dialogue = Dialogue(**config)
        response=dialogue.call_openai(prompt)
        print("*******************************************")
        print("Response from GPT:")
        print(response['content'])
        print("*******************************************\n")

def evaluate_on_GPT(sr3d_line_numbers):

    assert np.max(sr3d_line_numbers)<=65845,"line number %s > 65845!"%str(np.max(sr3d_line_numbers))
    assert np.min(sr3d_line_numbers)>=2,"line number %s < 2!"%str(np.max(sr3d_line_numbers))
    
    # 创建结果表格，格式如下
    # sr3d_line_number # scan_id # reference_type # target_id # answer_id # is_correct #
    sr3d_len=len(sr3d_line_numbers)
    results_table=np.zeros([sr3d_len,6],dtype='<U21')

    # # 创建dialogue实例
    # config = {
    #     'model': 'gpt-4',
    #     # 'model': 'gpt-3.5-turbo',
    #     'temperature': 0,
    #     'top_p': 0.1,
    #     'max_tokens': 'inf',
    #     'system_message': '',
    #     # 'load_path': 'chats/dialogue_an apple.json',
    #     'save_path': 'chats',
    #     'debug': False
    # }
    # dialogue = Dialogue(**config)

    # # 告知GPT背景信息
    # background_prompt=\
    # "I wiil describe some scenes and some objects in the scene, and I want you to analyse the spatial relationship of the objects in the scene and answer my questions." + \
    # " All descriptions are in right-handed Cartesian coordinate system with x-y-z axes, " + \
    # "where x represents left-right, y represents forward-backward, and z represents up-down. " + \
    # "In each scene, I will tell you the center position and size in x-y-z direction of the objects."
    # dialogue.call_openai(background_prompt)
    # print("--------------------------------------------")
    # print("background_prompot:\n"+background_prompt)
    # print("--------------------------------------------")

    # 记录时间作为文件名
    current_time = datetime.now()
    formatted_time = current_time.strftime("%Y-%m-%d-%H-%M-%S")

    # 遍历给定的数据集部分
    for idx,line_number in enumerate(sr3d_line_numbers):
        # 生成prompt
        prompt,info=gen_GPT_prompt_sr3d(line_number-1, SCANNET_DATA_ROOT, to_print=True, to_seperate_numbers=False)

        # 读取相关信息
        scan_id,target_id,distractor_ids,reference_type,utterance,csv_path=info

        # 打印并记录进度
        print("\n\nProcessing sr3d line %d, %d/%d."%(line_number,idx+1,sr3d_len))
        with open(SCRIPT_ROOT+"eval_results/%s-progress.log"%formatted_time,'a') as f:
            if idx==0:
                f.write(csv_path+'\n')
            f.write("Processing sr3d line %d, %d/%d.\n"%(line_number,idx+1,sr3d_len))
        

        # 获取GPT回复结果
        # dialogue = Dialogue(**config)
        # wrong_return_format=False
        # while True:
        #     try:
        #         response=dialogue.call_openai(prompt)["content"]
        #         response=call_openai(prompt)["content"]
        #         print("breaking while loop...")
        #         break
        #     except openai.error.RateLimitError as r:
        #         print("OpenAI RateLimitError!")
        #         print(r)
        #         time.sleep(1)
        #     except openai.error.ServiceUnavailableError as r:
        #         print("OpenAI ServiceUnavailableError!")
        #         print(r)
        #         time.sleep(1)
        #     except Exception as r:
        #         print("Something Unkown was wrong!")
        #         print(r)
        #         print("quit loop!")
        #         last_line="Something Unkown was wrong!"

        # 获取GPT回复结果（用了tenacity）
        response=call_openai(prompt)
        response=response['content']

        # 处理GPT的回复
        print("RESPOSE:\n",response)
        last_line = response.splitlines()[-1] if len(response) > 0 else ''
        print(type(last_line))
        print("last_line:",last_line)

        # 尝试读取last_line的长度（可能是None）
        try:
            length=len(last_line)
        except Exception as r:
            print(r)
            length=100
            with open(SCRIPT_ROOT+"eval_results/%s-progress.log"%formatted_time,'a') as f:
                f.write(r+'\n')
        
        # 如果没有按预期回复字典，则随机选一个
        wrong_return_format=False
        if length>12:
            answer_id=random.choice(np.append(distractor_ids,target_id))
            wrong_return_format=True
        else:
            # answer_id=response.splitlines()[-1]
            answer_id=eval(last_line)['ID']

        # 在表格中记录相关信息
        results_table[idx][0]=str(line_number)
        results_table[idx][1]=str(scan_id)
        results_table[idx][2]=str(reference_type)
        results_table[idx][3]=str(target_id)
        results_table[idx][4]=str(answer_id)
        if str(answer_id)==str(target_id):
            print("answer correct.")
            results_table[idx][5]=str(True)
            # 如果是错误返回格式，随后蒙对的，也要记录
            if wrong_return_format:
                log_error_info(line_number,prompt,response, SCRIPT_ROOT+"eval_results/%s.log"%formatted_time)
        else:
            print("answer wrong!")
            results_table[idx][5]=str(False)
            print("Error info:\nutterance: %s\ntarget_id:%s\nanswer_id:%s\nGPT response:%s"%(utterance,str(target_id),str(answer_id),response))
            # 记录错误信息
            log_error_info(line_number,prompt,response, SCRIPT_ROOT+"eval_results/%s.log"%formatted_time)

        # 保存结果表格
        save_path= SCRIPT_ROOT+"eval_results/%s.npy"%formatted_time
        np.save(save_path, results_table)
        print("results saved to: %s"%save_path)   
    
    return save_path

def log_error_info(line_number,prompt,GPT_response,log_file_path):
    error_info="------------------------------------------------------------\n"+\
    "LINE NUMBER: \n" + str(line_number) + "\n\n" + \
    "PROMPT: \n" + prompt + "\n\n" + \
    "GPT RESPONSE: \n" + GPT_response + "\n" + \
    "------------------------------------------------------------\n\n\n"

    with open(log_file_path,'a') as f:
        f.write(error_info)
        
    


def analyse_result(result_path):
    """
    分析保存好的结果npy文件
    # sr3d_line_number # scan_id # reference_type # target_id # answer_id # is_correct #
    """
    if isinstance(result_path,list):
        for idx,path in enumerate(result_path):
            result_single=np.load(path,allow_pickle=True)
            if not idx:
                result=result_single
            else:
                result=np.vstack([result,result_single])
    else:
        result=np.load(result_path,allow_pickle=True)
        
    print("Results for:",result_path)
    
    # 统计数据
    accuracy_count={
        "count_total":0,"correct_count_total":0,
        "count_horizontal":0,"correct_count_horizontal":0,
        "count_vertical":0,"correct_count_vertical":0,
        "count_support":0,"correct_count_support":0,
        "count_between":0,"correct_count_between":0,
        "count_allocentric":0,"correct_count_allocentric":0
    }
    for result_line in result:
        if result_line[0]=='':
            continue
        reference_type=result_line[2]
        accuracy_count["count_total"]+=1
        accuracy_count["count_"+reference_type]+=1
        if result_line[5]=="True":
            accuracy_count["correct_count_total"]+=1
            accuracy_count["correct_count_"+reference_type]+=1

    #分析正确率 
    for name in ["total","horizontal","vertical","support","between","allocentric"]:
        print(name+" accuracy:")
        correct=accuracy_count["correct_count_"+name]
        total=accuracy_count["count_"+name]
        percentage = "-" if total==0 else correct/total*100
        print(str(percentage)+"%% (%d/%d)"%(correct,total))
    
def random_sampling(lower, upper, mode, para):
    if mode=='rate':
        rate=para
        if not (0 < rate <= 1):
            raise ValueError("Rate must be a value between 0 (exclusive) and 1 (inclusive).")
        num_samples = int((upper - lower + 1) * rate)
    elif mode=='num':
        num_samples=para
    else:
        print("Invalid mode")
        return
    
    samples = random.sample(range(lower, upper + 1), num_samples)
    return samples



# gen_GPT_prompt_sr3d(57575,"/share/data/ripl/scannet_raw/train/") #第一个index是在excel中查看到的行数-1

# dialogue_with_GPT("/share/data/ripl/scannet_raw/train/")

# lines=np.arange(10000,10050) #sr3d中要测试的行数
# lines=np.arange(2,32)
# lines=random_sampling(2,2607,'num',30) # vertical
lines=random_sampling(2,3007,'num',30) # allocentric
result_path=evaluate_on_GPT(lines)
analyse_result(result_path)

# analyse_result(["./eval_results/2023-08-03-11-52-32.npy","./eval_results/2023-08-03-18-06-35.npy"])

# dialogue_with_GPT()
