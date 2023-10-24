# encoding:utf-8
import numpy as np
import csv,os,random,json,time
import re,ast
import logging,sys
from datetime import datetime
from copy import deepcopy
from code_interpreter import CodeInterpreter
from gpt_dialogue import Dialogue
from object_filter_gpt4 import ObjectFilter
from prompt_text import get_principle, get_principle_sr3d, get_system_message
from config import test_modes_nr3d,test_modes_sr3d,test_modes_scanrefer
from tenacity import (
    retry,
    before_sleep_log,
    stop_after_attempt,
    wait_exponential,
    wait_exponential_jitter,
    RetryError,
    
)  # for exponential backoff

# logging.basicConfig(stream=sys.stderr, level=logging.ERROR)
logger = logging.getLogger(__name__+'logger')
logger.setLevel(logging.ERROR)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.ERROR)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

class Refer3d:
    def __init__(self,scannet_data_root, script_root, dataset, refer_dataset_path, result_folder_name, gpt_config, scanrefer_iou_thr=0.5, use_gt_box=True, object_filter_result_check_folder_name=None, object_filter_result_check_list=None, use_principle=True,use_original_viewdep_judge=True,scanrefer_tool_name='mask3d',use_priority=False,use_code_interpreter=True) -> None:
        self.scannet_data_root=scannet_data_root
        self.script_root=script_root
        self.dataset=dataset
        self.refer_dataset_path=refer_dataset_path
        self.result_folder_name=result_folder_name
        self.gpt_config=gpt_config
        self.scanrefer_iou_thr=scanrefer_iou_thr
        self.use_gt_box=use_gt_box
        self.object_filter_result_check_folder_name=object_filter_result_check_folder_name
        self.object_filter_result_check_list=object_filter_result_check_list
        self.use_principle=use_principle
        self.use_original_viewdep_judge=use_original_viewdep_judge
        self.scanrefer_tool_name=scanrefer_tool_name
        self.use_priority=use_priority
        self.use_code_interpreter=use_code_interpreter
        self.token_usage_whole_run=0
        self.token_usage_this_ques=0
        self.time_consumed_whole_run=0
        self.time_consumed_this_ques=0

        self.raw_label_2_nyu40_idx=self.get_raw_label_2_nyu40_idx()
    
    def load_refer_dataset(self,line_numbers=[2,]):
        # load the refering dataset from the corresponding file, 
        # the dataset is one of (sr3d, nr3d, scanrefer).
        # and check if the line numbers is in available range.
        if self.dataset=='sr3d':
            self.sr3d_data=self.read_csv_with_index(self.refer_dataset_path)
            assert np.max(line_numbers)<=len(self.sr3d_data)+1,"line number %d > %d!"%(np.max(line_numbers),len(self.sr3d_data)+1)
            assert np.min(line_numbers)>=2,"sr3d line number %s < 2!"%np.min(line_numbers)
            return self.sr3d_data
        elif self.dataset=='nr3d':
            self.nr3d_data=self.read_csv_with_index(self.refer_dataset_path)
            assert np.max(line_numbers)<=len(self.nr3d_data)+1,"line number %d > %d!"%(np.max(line_numbers),len(self.nr3d_data)+1)
            assert np.min(line_numbers)>=2,"nr3d line number %s < 2!"%np.min(line_numbers)
            return self.nr3d_data
        elif self.dataset=='scanrefer':
            self.scanrefer_data=self.read_json(self.refer_dataset_path)
            assert np.max(line_numbers)<=len(self.scanrefer_data)-1,"line number %d > %d!"%(np.max(line_numbers),len(self.scanrefer_data)-1)
            assert np.min(line_numbers)>=0,"scanrefer description number %s < 0!"%np.min(line_numbers)
            return self.scanrefer_data
        else:
            print("Invalid dataset!")
            return None

    def get_scene_center(self,objects):
        xmin,ymin,zmin=float('inf'),float('inf'),float('inf')
        xmax,ymax,zmax=float('-inf'),float('-inf'),float('-inf')
        for obj in objects:
            x,y,z=obj['center_position']
            if x<xmin:
                xmin=x
            if x>xmax:
                xmax=x
            if y<ymin:
                ymin=y
            if y>ymax:
                ymax=y
            if z<zmin:
                zmin=z
            if z>zmax:
                zmax=z
        return self.round_list([(xmin+xmax)/2,(ymin+ymax)/2,(zmin+zmax)/2],2)        

    def round_list(self,lst,length):
        # round every element in lst
        for idx,num in enumerate(lst):
            lst[idx]=round(num,length)
        return list(lst)
    
    def get_scanrefer_gt_box(self,scan_id,object_id):
        # get the ground truth bounding box according to scan_id and object id
        # from file scan_id_aligned_bbox.npy, which could be produced in pre-process of ScanRefer repo.
        # scan_id_aligned_bbox.npy has matrices of shape (N, 8)，with each row as a box. box format is (cx,cy,cz,sx,sy,sz,label_id,obj_id).
        gt_box_path="/share/data/ripl/vincenttann/ScanRefer/data/scannet/scannet_data/"+scan_id+"_aligned_bbox.npy"
        gt_boxes=np.load(gt_box_path)
        gt_box=gt_boxes[ gt_boxes[:,-1].reshape(-1).astype(int)==int(object_id) ]
        assert len(gt_box)>0, "No gt box found!!! scan_id=%d, object_id=%d"%(scan_id,object_id)
        assert len(gt_box)==1, "Multiple gt box found!!! scan_id=%d, object_id=%d, gt_box found:%s"%(scan_id,object_id,str(gt_box))

        return self.center_size_to_extension(gt_box.reshape(-1)[0:6])
    
    def center_size_to_extension(self,box_center_size):
        cx,cy,cz,sx,sy,sz = box_center_size
        xmin=cx-sx/2
        xmax=cx+sx/2
        ymin=cy-sy/2
        ymax=cy+sy/2
        zmin=cz-sz/2
        zmax=cz+sz/2
        return [xmin,ymin,zmin,xmax,ymax,zmax]
    
    def extension_to_center_size(self,extension):
        xmin,ymin,zmin,xmax,ymax,zmax = extension
        cx=(xmin+xmax)/2
        cy=(ymin+ymax)/2
        cz=(zmin+zmax)/2
        sx=xmax-xmin
        sy=ymax-ymin
        sz=zmax-zmin
        return [cx,cy,cz,sx,sy,sz]
    
    def calc_iou(self,box1,box2):
        # format of boxes: (xmin,ymin,zmin,xmax,ymax,zmax)
        x1_min, y1_min, z1_min, x1_max, y1_max, z1_max = box1
        x2_min, y2_min, z2_min, x2_max, y2_max, z2_max = box2
        
        # itersection volume
        x_overlap = max(0, min(x1_max, x2_max) - max(x1_min, x2_min))
        y_overlap = max(0, min(y1_max, y2_max) - max(y1_min, y2_min))
        z_overlap = max(0, min(z1_max, z2_max) - max(z1_min, z2_min))
        intersection_volume = x_overlap * y_overlap * z_overlap
        
        # volume of two boxes
        volume_box1 = (x1_max - x1_min) * (y1_max - y1_min) * (z1_max - z1_min)
        volume_box2 = (x2_max - x2_min) * (y2_max - y2_min) * (z2_max - z2_min)
        
        # calculate IoU
        union_volume = volume_box1 + volume_box2 - intersection_volume
        iou = intersection_volume / union_volume if union_volume > 0 else 0.0
        
        return iou

    def non_max_suppression(self,objects_info_f, iou_threshold=0.5):
        print("before non_max_suppression: %d objects."%len(objects_info_f))
        # sort in order of conf score
        objects_info_f.sort(key=lambda x: x['score'], reverse=True)

        selected_objects = []

        # iterate through the list
        while len(objects_info_f) > 0:
            # take out the current object
            current_object = objects_info_f[0]
            selected_objects.append(current_object)
            objects_info_f.pop(0)

            # calculate iou with all other objects in list, delete those has higher iou than threshold.
            objects_info_f = [obj for obj in objects_info_f if self.calc_iou(current_object['extension'], obj['extension'])  < iou_threshold]
        
        print("after non_max_suppression: %d objects."%len(selected_objects))
        return selected_objects


    def read_csv_with_index(self,file_path):
        # read in the data of sr3d and nr3d(.csv)，return a dictionary.
        # use the line number of the csv file as index, start from 2.
        data = {}  
        with open(file_path, 'r', newline='', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            headers = next(reader)  # the first line is the header
            for index, row in enumerate(reader, start=2):  # interation start from line 2
                data[index] = dict(zip(headers, row))
        # print(len(data))
        return data
    
    def read_json(self,file_path):
        # read in the data of scanrefer(.json)，returning a list of dictionary(same as that in the json file). index starts from 0.
        with open(file_path, 'r') as jf:
            jf_data=jf.read() # jf_data is a string
            data=json.loads(jf_data)
        return data
    
    @retry(wait=wait_exponential_jitter(initial=20, max=120, jitter=20), stop=stop_after_attempt(5), before_sleep=before_sleep_log(logger,logging.ERROR)) #20s,40s,80s,120s + random.uniform(0,20)
    def get_gpt_response(self,prompt:str,code_interpreter:CodeInterpreter):
        # get response from GPT(using code interpreter). using retry from tenacity.
        # count the token usage and time as well
        # if the reponse does not include "Now the answer is complete", this means the answer is not done. attach an empty user message to let GPT to keep going.

        # start timing
        call_start_time=time.time()
        # the first call with the original prompt
        response,token_usage_total=code_interpreter.call_openai_with_code_interpreter(prompt)
        response=response['content']
        # loop until "Now the answer is complete" is in the response, or looping more than 10 times.
        count_response=0
        while not "Now the answer is complete" in response:
            if count_response >=10:
                print("Response does not end with 'Now the answer is complete.' !")
                break
            response,token_usage_add = code_interpreter.call_openai_with_code_interpreter('')
            response=response['content']
            token_usage_total+=token_usage_add
            count_response += 1
            print("count_response:",count_response)
        # stop timing
        call_end_time=time.time()
        time_consumed=call_end_time-call_start_time
        self.token_usage_this_ques+=token_usage_total
        self.token_usage_whole_run+=token_usage_total
        self.time_consumed_this_ques+=time_consumed
        self.time_consumed_whole_run+=time_consumed

        print("\n*** Refer model: token usage=%d, time consumed=%ds, TPM=%.2f ***"%(token_usage_total,time_consumed, token_usage_total/time_consumed*60))

        return response

    @retry(wait=wait_exponential_jitter(initial=20, max=120, jitter=20), stop=stop_after_attempt(5), before_sleep=before_sleep_log(logger,logging.ERROR)) #20s,40s,80s,120s + random.uniform(0,20)
    def get_gpt_response_no_code_interpreter(self,prompt:str,gpt_dialogue:Dialogue):
        # get response from GPT(without code interpreter). using retry from tenacity.
        # count the token usage and time as well
        # if the reponse does not include "Now the answer is complete", this means the answer is not done. attach an empty user message to let GPT to keep going.

        # start timing
        call_start_time=time.time()
        # firt call
        response,token_usage_total=gpt_dialogue.call_openai(prompt)
        response=response['content']
        # loop
        count_response=0
        while not "Now the answer is complete" in response:
            if count_response >=10:
                print("Response does not end with 'Now the answer is complete.' !")
                break
            response,token_usage_add = gpt_dialogue.call_openai('')
            response=response['content']
            token_usage_total+=token_usage_add
            # print('Bot:', response)
            count_response += 1
            print("count_response:",count_response)
        # stop timing
        call_end_time=time.time()
        time_consumed=call_end_time-call_start_time
        self.token_usage_this_ques+=token_usage_total
        self.token_usage_whole_run+=token_usage_total
        self.time_consumed_this_ques+=time_consumed
        self.time_consumed_whole_run+=time_consumed

        print("\n*** Refer model: token usage=%d, time consumed=%ds, TPM=%.2f ***"%(token_usage_total,time_consumed, token_usage_total/time_consumed*60))

        return response

    def scanrefer_answer_exist(self, data_index, iou_thr):
        # check whether it's possible to find the correct answer for the given index of scanrefer:
        # if we're using some bounding boxes detected by an object detector(like group-free) or instance segmentor(like mask3d), 
        # and if the largest IoU in IoUs of gt box and all boxes detected is less than the threshold(0.25 or 0.5), then it is impossible for the rest of our model to find the correct answer.
        # note the minimum of data_index is 0.
        data=self.scanrefer_data[data_index]
        scan_id=data['scene_id']
        target_id=data['object_id']
        target_class=data['object_name']
        utterance=data['description']
        annotation_id=data['ann_id']
        suffix='_'+self.scanrefer_tool_name if self.scanrefer_tool_name else ''
        npy_path_train=self.scannet_data_root + "/train/objects_info%s/objects_info%s_"%(suffix,suffix) + scan_id + ".npy" 
        npy_path_test=self.scannet_data_root+"/test/objects_info%s/objects_info%s_"%(suffix,suffix) +scan_id + ".npy"
        if os.path.exists(npy_path_train):
            npy_path=npy_path_train
        else:
            npy_path=npy_path_test
        # elif os.path.exists(npy_path_test):
        #     npy_path=npy_path_test
        # else:
        #     print("object_info.npy file does not exist!!! scan_id:",scan_id)
        objects_info=np.load(npy_path,allow_pickle=True)
        gt_box=self.get_scanrefer_gt_box(scan_id,target_id)
        iou_max=0.0
        iou_max_object=None
        for obj in objects_info:
            box=obj['extension']
            iou=self.calc_iou(gt_box,box)
            if iou>iou_max:
                iou_max=iou
                iou_max_object=obj
        info=(scan_id,target_id,target_class,utterance,annotation_id,gt_box,iou_max,iou_max_object)
        if iou_max>iou_thr:
            return True,info
        else:
            # print("No box has iou more than %.2f with gt box!!! iou_max is %.3f. Recorded to result and skipped."%(iou_thr,iou_max))
            return False,info
        
    def check_scanrefer_answer_exist_percentage(self,iou_thr):
        # check all data records in scanrefer and calculate the percentage that answer might exist, given the detected boxes.
        self.scanrefer_data=self.read_json(self.refer_dataset_path)
        answer_exist_count=0
        answer_exist_count_unique=0
        answer_exist_count_multiple=0
        total_unique=0
        total_multiple=0
        total=len(self.scanrefer_data)
        for idx in range(total):
            exist,_=self.scanrefer_answer_exist(idx,iou_thr)
            data=self.scanrefer_data[idx]
            answer_exist_count+=exist
            # 为unique and multiple
            is_unique=self.get_unique_info(data['scene_id'],data['object_name'])
            if is_unique:
                total_unique+=1
                answer_exist_count_unique+=exist
            else:
                total_multiple+=1
                answer_exist_count_multiple+=exist
                
        print(self.refer_dataset_path)

        percentage = -1 if total==0 else answer_exist_count/total*100
        print("answer exist cases(overall):")
        print("%.2f%% (%d/%d)"%(percentage,answer_exist_count,total))

        percentage = -1 if total_unique==0 else answer_exist_count_unique/total_unique*100
        print("answer exist cases(unique):")
        print("%.2f%% (%d/%d)"%(percentage,answer_exist_count_unique,total_unique))

        percentage = -1 if total_multiple==0 else answer_exist_count_multiple/total_multiple*100
        print("answer exist cases(multiple):")
        print("%.2f%% (%d/%d)"%(percentage,answer_exist_count_multiple,total_multiple))

    def find_relevant_objects(self,data_index,scan_id,target_id,utterance,npy_path,use_npy_file=True,object_info_list=None):
        # 新的两步法：先用object filter找到相关物体，在进行refer
        # 如果给出了object_filter_check_list，则在对应文件夹中检查，如果有则直接使用结果
        if self.object_filter_result_check_folder_name is not None:
            target_dialogue_name="%d_%s_%s_object_filter.json"%(data_index,scan_id,target_id)
            # 定义dialogue文件夹的路径
            folder_paths=["/share/data/ripl/vincenttann/sr3d/%s/%s/%s_dialogue_jsons/"%(self.object_filter_result_check_folder_name,f_time,f_time) for f_time in self.object_filter_result_check_list]
            # 遍历每个文件夹，检查是否包含目标文件
            found = False
            for folder_path in folder_paths:
                folder_contents = os.listdir(folder_path)
                if target_dialogue_name in folder_contents:
                    found = True
                    print(f"object filter dialogue '{target_dialogue_name}' found in '{folder_path}'.")
                    break
            if not found:
                print(f"object filter dialogue '{target_dialogue_name}' not found in check list. generate a new one.")
        else:
            found = False
        
        if found:
            target_dialogue_path=folder_path+target_dialogue_name
            with open(target_dialogue_path) as f:
                of_response=json.load(f)[-1]['content']
                last_line=of_response.splitlines()[-1]
            object_filter=ObjectFilter()
            relevant_ids=object_filter.extract_all_int_lists_from_text(last_line)
            
        else:
            target_dialogue_path=None
            object_filter=ObjectFilter()
            of_start_time=time.time()
            relevant_ids,token_usage_of=object_filter.filter_objects_by_description(description=utterance,use_npy_file=use_npy_file, objects_info_path=npy_path, object_info_list=object_info_list, to_print=True)

            # 统计时间和token
            of_end_time=time.time()
            time_consumed=of_end_time-of_start_time
            self.token_usage_this_ques+=token_usage_of
            self.token_usage_whole_run+=token_usage_of
            self.time_consumed_this_ques+=time_consumed
            self.time_consumed_whole_run+=time_consumed
            print("\n*** Object filter: token usage=%d, time consumed=%ds, TPM=%.2f ***\n"%(token_usage_of,time_consumed,token_usage_of/time_consumed*60))

        return relevant_ids,object_filter,target_dialogue_path

    def remove_spaces(self,s:str):
        return s.replace(' ','')
    
    def gen_prompt_compressed(self,data_index,to_print=True,deal_with_human_wrong_case=False,deal_with_not_mention_target_class=False):
        """
        对于sr3d/nr3d/scanrefer中的指定数据，返回compressed prompt以及其他相关信息
        """

        # 读取指定行的数据，如果数据中的correct_guess为FALSE则返回-1
        if self.dataset=='sr3d':
            data=self.sr3d_data[data_index]
        elif self.dataset=='nr3d':
            data=self.nr3d_data[data_index]
        else:
            data=self.scanrefer_data[data_index]
        if (self.dataset=='sr3d' or self.dataset=='nr3d') and (not deal_with_human_wrong_case) and (data['correct_guess'] in ['False','FALSE','false']):
            return -1,-1,-1
        if (self.dataset=='sr3d' or self.dataset=='nr3d') and (not deal_with_not_mention_target_class) and (data['mentions_target_class'] in ['False','FALSE','false']):
            return -2,-2,-2
        # 读入scan_id
        scan_id=data['scene_id'] if self.dataset=='scanrefer' else data['scan_id']
        if to_print:
            print("scan_id:",scan_id)

        # 读入refered class and object ids
        target_class=data['object_name'] if self.dataset=='scanrefer' else data["instance_type"]
        target_id=data['object_id'] if self.dataset=='scanrefer' else data["target_id"]

        # 读入utterance，根据情况补上句点
        utterance=data['description'] if self.dataset=='scanrefer' else data["utterance"]
        if not utterance.endswith("."):
            utterance += "."

        # 读入sr3d的reference type, distractors_ids, achor_types和anchor_ids
        if self.dataset=='sr3d':
            reference_type=data["coarse_reference_type"]
            distractor_ids=eval(data["distractor_ids"])
            anchor_classes=data["anchors_types"]
            anchor_ids=eval(data["anchor_ids"])

        # 读入nr3d 的一些信息
        elif self.dataset=='nr3d':
            mentions_target_class,uses_object_lang,uses_spatial_lang,uses_color_lang,uses_shape_lang=data["mentions_target_class"],data["uses_object_lang"],data["uses_spatial_lang"],data["uses_color_lang"],data["uses_shape_lang"]
        
        # 读入scanrefer的一些信息
        else:
            annotation_id=data['ann_id']

        # 读入事先准备好的物体信息，即npy文件
        npy_path_train=self.scannet_data_root + "/train/objects_info_%s/objects_info_%s_"%(self.scanrefer_tool_name,self.scanrefer_tool_name) + scan_id + ".npy" if (self.dataset=='scanrefer' and not self.use_gt_box) else self.scannet_data_root+"/train/objects_info/objects_info_"+scan_id+".npy"
        npy_path_test=self.scannet_data_root+"/test/objects_info_%s/objects_info_%s_"%(self.scanrefer_tool_name,self.scanrefer_tool_name) +scan_id + ".npy" if (self.dataset=='scanrefer' and not self.use_gt_box) else self.scannet_data_root+"/test/objects_info/objects_info_"+scan_id+".npy"
        if os.path.exists(npy_path_train):
            npy_path=npy_path_train
        elif os.path.exists(npy_path_test):
            npy_path=npy_path_test
        else:
            print("object_info.npy file does not exist!!! scan_id:",scan_id)
            return None, None, None
        objects_info=np.load(npy_path,allow_pickle=True) #objects_info是gt或3d segmentation得到的场景中所有物体的信息

        # 如果是scanrefer，要根据confidential score筛选一下
        if self.dataset=='scanrefer':
            objects_info_f=[]
            for obj in objects_info:
                score=obj.get('score')
                if score is None or score>0.4:
                    objects_info_f.append(obj)
            if score is not None:
                objects_info=self.non_max_suppression(objects_info_f)

        # 统计场景中所有物体的类别，用于scanrefer的unique/multiple分类
        # obj_idx_in_scene=[]
        # for obj in objects_info:
        #     obj_idx_in_scene.append(self.raw_label_2_nyu40_idx[obj['label']])
        # target_idx=self.raw_label_2_nyu40_idx[' '.join(target_class.split('_'))]
        # is_unique=True if obj_idx_in_scene.count(target_idx)<=1 else False
        is_unique=True

        # 如果是sr3d就只需要target，distractor和anchor，不用object filter
        if self.dataset=='sr3d':
            objects_related=[]
            anchor_has_front=True
            objects_related.append(objects_info[int(target_id)])
            for id in distractor_ids:
                objects_related.append(objects_info[int(id)])
            for id in anchor_ids:
                objects_related.append(objects_info[int(id)])
                anchor_has_front=anchor_has_front and objects_info[int(id)]['has_front']
                
        else:
            # relevant_ids,object_filter,target_dialogue_path=self.find_relevant_objects(data_index,scan_id,target_id,utterance,npy_path)
            relevant_ids,object_filter,target_dialogue_path=self.find_relevant_objects(data_index,scan_id,target_id,utterance,npy_path,use_npy_file=False,object_info_list=objects_info)

            # objects_related = objects_info if (relevant_ids is None) else objects_info[relevant_ids]
            objects_related = objects_info if (relevant_ids is None) else [obj for obj in objects_info if obj['id'] in relevant_ids]

        # # 对于sr3d记录anchor_has_front
        # if self.dataset=='sr3d':
        #     anchor_has_front=True
        #     for id in anchor_ids:
        #         anchor_has_front=anchor_has_front and objects_info[int(id)]['has_front']

        # 获取场景的中心坐标
        # scene_center=self.get_scene_center(objects_related)
        scene_center=self.get_scene_center(objects_info) # 注意这里应该用所有物体的信息，而不只是relevant

        # 生成prompt中的背景信息部分
        prompt=scan_id + ":objs with quant description based on r-h Cartesian coord sys with x-y-z axes,  x-y plane=ground, z-axis=up/down. coords format [x, y, z].\n" 
        if not self.dataset=='sr3d':
            prompt=prompt+"Scene center:%s. If no direction vector, observer at center for obj orientation.\n"%self.remove_spaces(str(scene_center))
        
        prompt=prompt+"objs list:\n"

        # 生成prompt中对物体的定量描述部分（遍历所有相关物体）
        for obj in objects_related:
            # 位置信息，保留2位小数
            center_position=obj['center_position']
            center_position=self.round_list(center_position,2)
            # size信息，保留2位小数
            size=obj['size']
            size=self.round_list(size,2)
            # extension信息，保留2位小数
            extension=obj['extension']
            extension=self.round_list(extension,2)
            # 方向信息，用方向向量表示
            if obj['has_front']:
                front_point=np.array(obj['front_point'])
                center=np.array(obj['obb'][0:3])
                direction_vector=front_point-center
                direction_vector_normalized=direction_vector/np.linalg.norm(direction_vector)
                # 再计算左和右的方向向量，全部保留两位小数
                front_vector=self.round_list(direction_vector_normalized,2)
                up_vector=np.array([0,0,1])
                left_vector=self.round_list(np.cross(direction_vector_normalized,up_vector),2)
                right_vector=self.round_list(np.cross(up_vector,direction_vector_normalized),2)
                behind_vector=self.round_list(-np.array(front_vector),2)
                # 生成方向信息                
                # direction_info="The direction that this %s is facing can be represented by a normalized direction vector %s\n"%(obj['label'], direction_vector_normalized)
                direction_info=";direction vectors:front=%s,left=%s,right=%s,behind=%s\n"%(front_vector,left_vector,right_vector,behind_vector)
                # 
            else:
                # direction_vector=None
                # direction_vector_normalized=None
                # direction_info="The direction that this %s is facing is unknown.\n"%obj['label']
                direction_info="\n" #未知方向向量就啥都不写


            # sr3d，给出center、size
            if self.dataset=='sr3d':
                line="%s,id=%s,ctr=%s,size=%s" %(obj['label'], obj['id'], self.remove_spaces(str(center_position)), self.remove_spaces(str(size)) )

            # nr3d和scanrefer，给出center、size、color
            else:
                color=obj['median_rgba'][0:3] if (self.dataset=='scanrefer' and not self.use_gt_box) else obj['avg_rgba'][0:3]
                line="%s,id=%s,ctr=%s,size=%s,RGB=%s" %(obj['label'], obj['id'], self.remove_spaces(str(center_position)), self.remove_spaces(str(size)), self.remove_spaces(str(color) ))
                
            prompt=prompt+line+direction_info


        # prompt中的要求
        line="Instruction:find the one described object in description: \n\"%s\"\n" %utterance
        prompt=prompt+line
        if self.dataset=='sr3d':
            prompt=prompt+get_principle_sr3d(utterance) if self.use_principle else prompt
        else:
            prompt=prompt+get_principle(utterance,self.use_priority) if self.use_principle else prompt
        # if not self.dataset=='sr3d':
        #     # prompt=prompt+" Howerver, if the direction vector of A is not provided, you should use other information to identify the referred object instead of assuming a direction vector."
            
        prompt=prompt+"\nThere is exactly one answer, so if you receive multiple answers, consider other constraints; if get no answers, loosen constraints."
        prompt=prompt+"\nWork this out step by step to ensure right answer."
        # prompt=prompt+"\nYou should calculate the result in each step and tell me the exact final result."
        # prompt=prompt+"But Do Not response anything except the id."
        # prompt=prompt+"\nIn the last line of your response, there should Only be a python dictionary in format: {'ID':id}, where id is the id of the referred object."

        # prompt=prompt+"\nIf the answewr is complete, add 'Now the answer is complete.' to the end of your answer."
        # prompt=prompt+"\n Then in the last line of your response, there should Only be a python dictionary in format: {'ID':id}, where id is the id of the referred object."

        prompt=prompt+"\nIf the answer is complete, add \"Now the answer is complete -- {'ID':id}\" to the end of your answer(that is, your completion, not your code), where id is the id of the referred obj."

        # prompt=prompt+"\n Do not stop before you find the right id."

        if to_print:
            print("--------------------------------------------")
            print("Generated prompt:\n"+prompt)
            print("--------------------------------------------")
            print("Right answer:",target_id)
            print("")
        if self.dataset=='sr3d':
            relevant_ids=None
            info=(scan_id,target_id,target_class,distractor_ids,reference_type,utterance,anchor_has_front)
        elif self.dataset=='nr3d':
            info=(scan_id,target_id,target_class,utterance,mentions_target_class,uses_object_lang,uses_spatial_lang,uses_color_lang,uses_shape_lang,object_filter,target_dialogue_path)
        else:
            gt_box=self.get_scanrefer_gt_box(scan_id, target_id)
            info=(scan_id,target_id,target_class,utterance,annotation_id,objects_related,gt_box,object_filter,target_dialogue_path,is_unique)
    

        return prompt,info,relevant_ids
    
    def extract_answer_id_from_last_line(self,last_line,random_choice_list=[0,]):
        # 如果没有按照预期格式回复则随机选取(Sr3d)或直接选成0(Nr3d和Scanrefer);按预期格式恢复则提取答案
        wrong_return_format=False
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
                # 如果确实以 Now the answer is complete -- {'ID': xxx} 的格式回复了，但是xxx不是数字（例如是None），也能随机选。
                if not isinstance(answer_id,int):
                    if isinstance(answer_id,list) and all([isinstance(e,int) for e in answer_id]):
                        print("Wrong answer format: %s. random choice from this list"%str(answer_id))
                        answer_id=random.choice(answer_id)
                    else:
                        print("Wrong answer format: %s. No dict found. Random choice from relevant objects."%str(answer_id))
                        answer_id=random.choice(random_choice_list)
                    wrong_return_format=True
            except:
                print("Wrong answer format!! No dict found. Random choice.")
                answer_id=random.choice(random_choice_list)
                wrong_return_format=True
        else:
            print("Wrong answer format!! No dict found. Random choice.")
            answer_id=random.choice(random_choice_list)
            wrong_return_format=True
        
        return answer_id,wrong_return_format
        
    def evaluate_on_GPT(self, line_numbers):
        """
        @descr  the most important function. run evluation for the given data records decided by the line_numbers.  then save the result table to npy file.
        @param  line_numbers: a list of data record indices. for sr3d and nr3d, the minimum is 2. for scanrefer, it's 0.
        """
        # first load the refering dataset.
        self.load_refer_dataset(line_numbers)

        # create a table for recording results. format:
        #       0     #     1   #       2        #     3     #     4     #      5     #          6            #         7
        # sr3d: 
        # line_number # scan_id # reference_type # target_id # answer_id # is_correct #  anchors_has_front    #
        # nr3d:
        # line_number # scan_id #    None        # target_id # answer_id # is_correct # mentions_target_class # uses_object_lang # uses_spatial_lang # uses_color_lang # uses_shape_lang
        # scanrefer:
        #  dscrp_num  # scan_id #    ann_id      # target_id # answer_id #   gt_box   #     answer_box        #     iou          #  object_class      # correct_answer_exist # iou_max   # is_unique

        dataset_len= len(line_numbers)
        results_table=np.zeros([dataset_len,12],dtype='<U21')

        # record current time for the name of the files.
        current_time = datetime.now()
        formatted_time = current_time.strftime("%Y-%m-%d-%H-%M-%S")
        print("formatted_time:",formatted_time)

        # create a result folder for the chosen test mode if it does not exist.
        if not os.path.exists(self.script_root+self.result_folder_name):
            os.makedirs(self.script_root+self.result_folder_name)

        # the subfolder of the current experiment. named after the time.
        results_sub_folder=self.script_root+self.result_folder_name+formatted_time+'/' 
        if not os.path.exists(results_sub_folder):
            os.makedirs(results_sub_folder)

        # path of relevant files.
        process_log_file=results_sub_folder+"%s-progress.log"%formatted_time
        success_log_file=results_sub_folder+"%s-success.log"%formatted_time
        failure_log_file=results_sub_folder+"%s-failure.log"%formatted_time
        result_npy_file=results_sub_folder+"%s.npy"%formatted_time
        dialogue_json_folder=results_sub_folder+"%s_dialogue_jsons/"%formatted_time
        os.makedirs(dialogue_json_folder)

        # iterate through the chosen part of dataset
        for idx,line_number in enumerate(line_numbers):
            # print and record the process
            print("\n\nProcessing %s line %d, %d/%d."%(self.dataset,line_number,idx+1,dataset_len))
            
            with open(process_log_file,'a') as f:
                if idx==0:
                    f.write(self.refer_dataset_path+'\n')
                    f.write(str(list(line_numbers))+'\n')
                f.write("\nProcessing %s line %d, %d/%d. "%(self.dataset,line_number,idx+1,dataset_len))

            # for scanrefer, check if answer might exist. if not, record this and save.
            if self.dataset=='scanrefer':
                exist,info=self.scanrefer_answer_exist(line_number,iou_thr=0.25)
                scan_id,target_id,target_class,utterance,annotation_id,gt_box,iou_max,iou_max_object=info
                results_table[idx][9]=exist #correct_answer_exist
                results_table[idx][10]=iou_max #iou_max
                with open(process_log_file,'a') as f:
                    f.write("iou_max=%.3f. "%iou_max)
                if not exist and not self.use_gt_box:                    
                    results_table[idx][0]=line_number
                    results_table[idx][1]=scan_id
                    results_table[idx][2]=annotation_id
                    results_table[idx][3]=target_id
                    results_table[idx][5]=str(list(gt_box))
                    results_table[idx][6]=None #answer box
                    results_table[idx][7]=0.0 #iou
                    results_table[idx][8]=target_class
                    results_table[idx][10]=iou_max
                    # results_table[idx][11]=is_unique
                    with open(process_log_file,'a') as f:
                        f.write("No correct answer, iou_max is %.3f, under 0.25, Skipped."%iou_max)
                    np.save(result_npy_file, results_table)
                    print("results saved to: %s\n\n"%result_npy_file) 
                    continue
            
            # token and time usage
            self.time_consumed_this_ques=0
            self.token_usage_this_ques=0

            # generate prompt
            prompt,info,relevant_ids=self.gen_prompt_compressed(line_number, to_print=True)
            if prompt is None:
                with open(process_log_file,'a') as f:
                    f.write("prompt not generated. Perhaps the object_info npy file does not exist.")
                continue

            elif prompt==-1:
                with open(process_log_file,'a') as f:
                    f.write("Human failed to find this object: '%s'. Skipped."%utterance)
                continue

            elif prompt==-2:
                with open(process_log_file,'a') as f:
                    f.write("Not mention target class '%s'. Skipped."%utterance)
                continue

            # read some information from info
            if self.dataset=='sr3d':
                scan_id,target_id,target_class,distractor_ids,reference_type,utterance,anchor_has_front=info
                object_filter=ObjectFilter()
                prev_of_dialogue_path=None
            elif self.dataset=='nr3d':
                scan_id,target_id,target_class,utterance, mentions_target_class,uses_object_lang,uses_spatial_lang,uses_color_lang,uses_shape_lang,object_filter,prev_of_dialogue_path = info
            else:
                scan_id,target_id,target_class,utterance,annotation_id,objects_related,gt_box,object_filter,prev_of_dialogue_path,is_unique=info
            object_filter: ObjectFilter
            
            # 尝试获取GPT回复。如果出现Retry Error，那就last_line随便设置，最终导致wrong_format=True，随机选取
            get_gpt_response_success=True
            try:
                if self.use_code_interpreter:
                    code_interpreter=CodeInterpreter(**self.gpt_config)
                    response=self.get_gpt_response(prompt,code_interpreter)
                else:
                    gpt_dialogue=Dialogue(**self.gpt_config)
                    response=self.get_gpt_response_no_code_interpreter(prompt,gpt_dialogue)
                    code_interpreter=gpt_dialogue # 这里必须给code_interpreter绑定一个值
                print("\n*** This question: token usage=%d, time consumed=%ss, TPM=%.2f ***"%(self.token_usage_this_ques, self.time_consumed_this_ques, self.token_usage_this_ques/self.time_consumed_this_ques*60))
                print("*** Whole run: token usage=%d, time consumed=%ss, TPM=%.2f ***\n"%(self.token_usage_whole_run, self.time_consumed_whole_run, self.token_usage_whole_run/self.time_consumed_whole_run*60))
            except RetryError as r:
                print(r)
                with open(process_log_file,'a') as f:
                    f.write("ReTry Error.")
                response="Fail to get response from GPT. RetryError in func get_gpt_response"
                last_line="Nonesense"
                get_gpt_response_success=False

            # 处理GPT的回复 （如果成功获取）
            if get_gpt_response_success:
                print("--------------------------------------------")
                print("DIALOGUE:")
                code_interpreter.print_pretext()
                print("--------------------------------------------")
                last_line=response.splitlines()[-1]
                print(type(last_line))
                print("last_line:",last_line)

            # 从last_line中获取answer_id，如果格式不符合要求则从relevant_ids中随机选取
            random_choice_list=np.append(distractor_ids,target_id) if self.dataset=='sr3d' else relevant_ids
            answer_id, wrong_return_format=self.extract_answer_id_from_last_line(last_line,random_choice_list)

            # 对于scanrefer，要找到answer_id对应的box并计算iou
            if self.dataset=='scanrefer':
                for obj in objects_related:
                    if obj['id']==answer_id:
                        answer_object=obj
                        break
                # answer_object=objects_related[answer_id]
                answer_box=self.center_size_to_extension(np.append(answer_object['center_position'],answer_object['size']))
                iou=self.calc_iou(answer_box,gt_box)

            # 在表格中记录相关信息
            results_table[idx][0]=line_number
            results_table[idx][1]=scan_id
            results_table[idx][3]=target_id
            results_table[idx][4]=answer_id
            if self.dataset=='sr3d':
                results_table[idx][2]=reference_type
                results_table[idx][6]=anchor_has_front
            elif self.dataset=='nr3d':
                results_table[idx][2]='None'
                results_table[idx][6]= mentions_target_class
                results_table[idx][7]= uses_object_lang 
                results_table[idx][8]= uses_spatial_lang
                results_table[idx][9]= uses_color_lang
                results_table[idx][10]= uses_shape_lang
            else:
                results_table[idx][2]=annotation_id
                results_table[idx][5]=str(list(gt_box))
                results_table[idx][6]=str(list(answer_box))
                results_table[idx][7]=iou
                results_table[idx][8]=target_class
                # results_table[idx][10]=is_unique

            # update 'printed_pretext' for code_interpreter and object_filter
            code_interpreter.print_pretext(to_print_out=False)
            object_filter.print_pretext(to_print_out=False)

            # 对于sr3d和nr3d，比较answer_id和target_id来判断是否回答正确
            if self.dataset=='sr3d' or self.dataset=='nr3d':
                if str(answer_id)==str(target_id):
                    answer_correct=True
                    print("answer correct.")
                    results_table[idx][5]=True
                    #记录正确信息
                    self.log_info(line_number, scan_id, utterance, object_filter.printed_pretext, code_interpreter.printed_pretext, success_log_file, target_id, answer_id)
                    with open(process_log_file,'a') as f:
                        f.write("answer correct.")
                    # 如果是错误返回格式，随后蒙对的，也要记录在错误log中
                    if wrong_return_format:
                        self.log_info(line_number, scan_id, utterance, object_filter.printed_pretext, code_interpreter.printed_pretext, failure_log_file, target_id, answer_id)
                        with open(process_log_file,'a') as f:
                            f.write("But it's a guess after receiving wrong format.")
                else:
                    answer_correct=False
                    print("answer wrong!")
                    results_table[idx][5]=str(False)
                    print("Error info:\nutterance: %s\ntarget_id:%s\nanswer_id:%s\nGPT last response:%s"%(utterance,str(target_id),str(answer_id),response))
                    # 记录错误信息
                    self.log_info(line_number, scan_id, utterance, object_filter.printed_pretext, code_interpreter.printed_pretext, failure_log_file, target_id, answer_id)
                    with open(process_log_file,'a') as f:
                        f.write("answer wrong!")

            # 对于scanrefer，按iou是否超过阈值来判断
            else:
                target_id_text=str(target_id) + "(ScanNet) / "+str(iou_max_object['id']) + "(GroupFree)"
                if iou>self.scanrefer_iou_thr:
                    answer_correct=True
                    print("answer correct: IoU=%.3f"%iou)
                    #记录正确信息
                    self.log_info(line_number, scan_id, utterance, object_filter.printed_pretext, code_interpreter.printed_pretext, success_log_file, target_id_text, answer_id, iou, iou_max)
                    with open(process_log_file,'a') as f:
                        f.write("answer correct. iou=%.3f"%iou)
                else:
                    answer_correct=False
                    print("answer wrong! IoU=%.3f"%iou)
                    # 记录错误信息
                    self.log_info(line_number, scan_id, utterance, object_filter.printed_pretext, code_interpreter.printed_pretext, failure_log_file, target_id_text, answer_id, iou, iou_max)
                    with open(process_log_file,'a') as f:
                        f.write("answer wrong! iou=%.3f"%iou)

            # 保存对话到json文件
            if prev_of_dialogue_path:
                import shutil
                shutil.copy(prev_of_dialogue_path,dialogue_json_folder)
                print("copy previous object filter dialogue %s to %s"%(prev_of_dialogue_path,dialogue_json_folder))
            else:
                object_filter_json_name="%d_%s_%s_object_filter.json"%(line_number,scan_id,target_id)
                object_filter.save_pretext(dialogue_json_folder,object_filter_json_name)
            success_text="success" if (answer_correct and not wrong_return_format) else "failure"
            refer_json_name="%d_%s_%s_refer_%s.json"%(line_number,scan_id,target_id,success_text)
            code_interpreter.save_pretext(dialogue_json_folder,refer_json_name)

            # 保存结果表格
            np.save(result_npy_file, results_table)
            print("results saved to: %s\n\n"%result_npy_file)   

        self.save_path=result_npy_file

        return formatted_time

    def self_correction(self,failure_diagolue_path,target_id,target_class):
        # 读入failure dialogue备用
        with open(failure_diagolue_path,'r') as f:
            failure_dialogue=json.load(f)
            failure_dialogue_length=len(failure_dialogue)
            # original_user_dialogue=failure_dialogue[0:2] # system and user

        # 初始化code interpreter
        gpt_config=deepcopy(self.gpt_config)
        gpt_config['load_path']=failure_diagolue_path
        code_interpreter=CodeInterpreter(**gpt_config)
        code_interpreter.print_pretext()

        # 准备prompt并让gpt自行发现问题，直到其输出Now the answer has complete
        print("\nself correcting...\n")
        correction_prompt="The correct answer is %s %d. Can you double check the information of %s %d and the given prompt and see where you got wrong? Still, add \"Now the answer is complete -- {'ID':id}\" to the end of your answer, where id is the correct id of the referred obj."%(target_class,int(target_id),target_class,int(target_id))
        print("correctin prompt:",correction_prompt)
        self.get_gpt_response(correction_prompt,code_interpreter)
        print("--------------------------------------------")
        print("ORIGINAL PROMPT AND CORRECTION DIALOGUE:")
        code_interpreter.print_pretext(print_system_and_user_first_prompt=False)
        print("--------------------------------------------")
        self_correction_length=len(code_interpreter.pretext)-failure_dialogue_length #self correction新增的长度
        
        # 删除gpt之前的错误推理，并让其完整输出推理过程
        print("\nregenerating reasoning process...\n")
        del code_interpreter.pretext[2:failure_dialogue_length]
        regenerate_prompt="Now you have the correct reasoning and result. Can you generate the whole reasonging process to get this correct answer from the very beginning? You cannot use the code execution result above and have to generate code when needed. When answer step by step, stop whenever you feel there is need to generate python code and wait for the result from the code execution. Remember to use print() function to print out the result and keep two decimal places for numbers."
        print("regenerate prompt:",regenerate_prompt)
        response=self.get_gpt_response(regenerate_prompt,code_interpreter)
        print("--------------------------------------------")
        print("RE-GENERATED REASONING DIALOGUE:")
        code_interpreter.print_pretext(print_system_and_user_first_prompt=False)
        print("--------------------------------------------")

        # 提取结果并检查是否为正确答案
        last_line=response.splitlines()[-1]
        answer_id,_=self.extract_answer_id_from_last_line(last_line)
        if str(answer_id)==str(target_id):
            # correction后答案正确，删除correction prompt部分，只保留original prompt和推理过程
            del code_interpreter.pretext[2:2+self_correction_length]
            correction_success=True
        else:
            print("wrong answer id after correction!!")
            correction_success=False
        
        return code_interpreter,correction_success

    def self_correction_dataset(self,result_folder_path,formatted_time,line_number_list):
        # 首先确定refer数据集
        refer_dataset=self.load_refer_dataset()
        
        # 定义dialogue文件夹路径
        dialogue_folder_path="%s%s/%s_dialogue_jsons/" % (result_folder_path, formatted_time, formatted_time)

        # 遍历指定line_number_list
        for line_number in line_number_list:
            # 获取相关数据
            data_line=refer_dataset[line_number]
            scan_id=data_line['scene_id'] if self.dataset=='scanrefer' else data_line['scan_id']
            target_id=data_line['object_id'] if self.dataset=='scanrefer' else data_line['target_id']
            target_class=data_line['object_name'] if self.dataset=='scanrefer' else data_line['instance_type']
            # 定义原始failure dialogue的路径
            dialogue_path=dialogue_folder_path + "%d_%s_%s_refer_failure.json"%(line_number,scan_id,target_id)
            # correction dialogue的路径
            correction_dialogue_name="%d_%s_%s_refer_correction.json"%(line_number,scan_id,target_id)
            correction_dialogue_path=dialogue_folder_path+correction_dialogue_name
            # 检查correction dialogue是否存在，如果已经存在则跳过
            if os.path.exists(correction_dialogue_path):
                print("correction dialogue %s already exists! skipped."%correction_dialogue_path)
                continue
            # 如果failure dialogue存在（说明是错误案例），则改正后保存到新文件
            if os.path.exists(dialogue_path):
                print("failure dialogue found: "+dialogue_path)
                try:
                    code_interpreter,correction_success=self.self_correction(dialogue_path,target_id,target_class)
                except Exception as e:
                    print("exception arised!!!")
                    print(e)
                    code_interpreter=CodeInterpreter()
                    correction_success=False

                if correction_success:
                    code_interpreter.save_pretext(dialogue_folder_path,correction_dialogue_name)
                    print("correction succeed! saved to: %s%s"% (dialogue_folder_path,correction_dialogue_name))
                else:
                    correction_dialogue_name="%d_%s_%s_refer_correction_fail.json"%(line_number,scan_id,target_id)
                    code_interpreter.save_pretext(dialogue_folder_path,correction_dialogue_name)
                    print("correction fail! saved to: %s%s"% (dialogue_folder_path,correction_dialogue_name))
            # 如果不存在（说明是正确案例或line_number不正确），则跳过
            else:
                print("failure dialogue not found! "+dialogue_path)
    
    def log_info(self,line_number,scan_id,utterance,dialogue_object_filter,dialogue_refer,log_file_path, correct_id, answer_id, iou=None,max_iou=None):
        info="------------------------------------------------------------\n"
        info=info+"LINE NUMBER: \n" + str(line_number) + "\n\n" 
        info=info+"SCAN ID: \n" + scan_id + "\n\n"
        info=info+"UTTERANCE: \n" + utterance + "\n\n"
        info=info+"CORRECT ID: \n" + str(correct_id) + "\n\n" 
        info=info+"ANSWER ID: \n" + str(answer_id) + "\n\n"
        if not (iou is None):
            info=info+"IoU:\n%.3f\n\n"%iou
            info=info+"MAX IoU:\n%.3f\n\n"%max_iou
        info=info+"DIALOGUE OBJECT FILTER: \n" + dialogue_object_filter + "\n"
        info=info+"DIALOGUE REFER: \n" + dialogue_refer + "\n" + \
        "------------------------------------------------------------\n\n\n"

        with open(log_file_path,'a') as f:
            f.write(info)

    def get_easy_info(self,line_number)->bool:
        # 对于sr3d和nr3d，检查line_number对应数据的难度。
        # 同类物体（包括正确物体自身）<=2个则为easy，否则为hard
        if self.dataset=='sr3d':
            refer_data=self.sr3d_data[int(line_number)]
            distractor_ids=eval(refer_data['distractor_ids'])
            # print(distractor_ids)
            is_easy=True if len(distractor_ids)<=1 else False
        else:
            refer_data=self.nr3d_data[int(line_number)]
            # nr3d的stimulus_id格式: scan_id-target_class-target_id-distractor_id1-...-distractor_idn
            stimulus_id=refer_data['stimulus_id']
            n_object_same_class = int(stimulus_id.split('-')[2])
            is_easy=True if n_object_same_class<=2 else False
        return is_easy
    
    def get_view_dep_info(self,line_number)->bool:
        # 对于sr3d和nr3d，检查utterance是否为view_dependent.对照了referit3d和butd的代码，只需要检查utterance中是否出现以下关键词
        refer_data=self.sr3d_data[int(line_number)] if self.dataset=='sr3d' else self.nr3d_data[int(line_number)]
        utterance=refer_data['utterance']
        rels = [
            'front', 'behind', 'back', 'left', 'right', 'facing',
            'leftmost', 'rightmost', 'looking', 'across'
        ]
        if self.use_original_viewdep_judge:
            words = set(utterance.split()) # ... on the left.
            return any(rel in words for rel in rels)
        else:
            return any(rel in utterance for rel in rels)
        
    def get_left_right_info(self,line_number)->bool:
        refer_data=self.sr3d_data[int(line_number)] if self.dataset=='sr3d' else self.nr3d_data[int(line_number)]
        utterance=refer_data['utterance']
        rels = [
            'left', 'right',
            'leftmost', 'rightmost'
        ]
        if self.use_original_viewdep_judge:
            words = set(utterance.split())
            return any(rel in words for rel in rels)
        else:
            return any(rel in utterance for rel in rels)

    def get_ordinal_info(self,line_number)->bool:
        refer_data=self.sr3d_data[int(line_number)] if self.dataset=='sr3d' else self.nr3d_data[int(line_number)]
        utterance=refer_data['utterance']
        rels = [
            'from left', 'from right',
            'from the left', 'from the right'
        ]
        # words = set(utterance.split())
        return any(rel in utterance for rel in rels)

    def get_correct_guess_info(self,line_number)->bool:
        # print(self.nr3d_data.keys())
        refer_data=self.nr3d_data[int(line_number)]
        if refer_data['correct_guess'] in ['True','TRUE','true']:
            return True
        else:
            return False
    
    def analyse_result_sr3d(self,result_path):
        # 本函数用于分析nr3d的结果
        # 首先处理path，如果是列表，就把所有的npy文件对应的np array合并
        if isinstance(result_path,list):
            for idx,path in enumerate(result_path):
                result_single=np.load(path,allow_pickle=True)
                if not idx:
                    result=result_single
                else:
                    result=np.vstack([result,result_single])
        else:
            result=np.load(result_path,allow_pickle=True)
        print("Sr3d results for:",result_path)
        # result=result[0:110,:]
        # 定义记录结果的字典
        accuracy_count={
            "count_total":0,"correct_count_total":0,
            "count_easy":0,"correct_count_easy":0,
            "count_hard":0,"correct_count_hard":0,
            "count_view_dep":0,"correct_count_view_dep":0,
            "count_view_indep":0,"correct_count_view_indep":0,
            "count_left_right":0,"correct_count_left_right":0,
            "count_horizontal":0,"correct_count_horizontal":0,
            "count_vertical":0,"correct_count_vertical":0,
            "count_support":0,"correct_count_support":0,
            "count_between":0,"correct_count_between":0,
            "count_allocentric":0,"correct_count_allocentric":0
        }

        # 遍历，统计结果
        wrong_line_numbers=[]
        for result_line in result:
            # 首先读取line_number
            line_number=result_line[0] #注意这里读进来是str
            # 如果是空行则跳过
            if result_line[0]=='':
                continue
            # 总数记数
            accuracy_count["count_total"]+=1
            # 获取easy信息并给easy的总数记数
            is_easy=self.get_easy_info(line_number)
            easy_setting = 'easy' if is_easy else 'hard'
            accuracy_count['count_%s'%easy_setting]+=1
            # 获取view_dependent信息并记数
            is_view_dep=self.get_view_dep_info(line_number)
            view_dep_setting = 'view_dep' if is_view_dep else 'view_indep'
            accuracy_count['count_%s'%view_dep_setting]+=1
            # 获取left_right信息并记数
            has_left_right=self.get_left_right_info(line_number)
            accuracy_count['count_left_right']+=1 if has_left_right else 0
            # 五类空间关系记数
            reference_type=result_line[2]
            accuracy_count["count_"+reference_type]+=1

            # 给正确案例记数
            if result_line[5]=="True":
                accuracy_count["correct_count_total"]+=1
                accuracy_count['correct_count_%s'%easy_setting]+=1
                accuracy_count['correct_count_%s'%view_dep_setting]+=1 
                accuracy_count['correct_count_left_right']+=1 if has_left_right else 0
                accuracy_count["correct_count_"+reference_type]+=1

            else:
                wrong_line_numbers.append(eval(result_line[0]))
        
        # 打印准确率
        for name in ['total','easy','hard','view_dep','view_indep','left_right','horizontal','vertical','support','between','allocentric']:
            print(name+" accuracy:")
            correct=accuracy_count["correct_count_"+name]
            total=accuracy_count["count_"+name]
            percentage = -1 if total==0 else correct/total*100
            print("%.2f%% (%d/%d)"%(percentage,correct,total))

    def analyse_result_nr3d(self,result_path,skip_human_wrong_cases=True):
        # 本函数用于分析nr3d的结果
        # 首先处理path，如果是列表，就把所有的npy文件对应的np array合并
        if isinstance(result_path,list):
            for idx,path in enumerate(result_path):
                result_single=np.load(path,allow_pickle=True)
                if not idx:
                    result=result_single
                else:
                    result=np.vstack([result,result_single])
        else:
            result=np.load(result_path,allow_pickle=True)
        print("Nr3d results for:",result_path)
        # result=result[0:110,:]
        # 定义记录结果的字典
        accuracy_count={
            "count_total":0,"correct_count_total":0,
            "count_easy":0,"correct_count_easy":0,
            "count_hard":0,"correct_count_hard":0,
            "count_view_dep":0,"correct_count_view_dep":0,
            "count_view_indep":0,"correct_count_view_indep":0,
            "count_left_right":0,"correct_count_left_right":0,
            "count_ordinal":0,"correct_count_ordinal":0, # from the left/right
            "count_use_object":0,"correct_count_use_object":0,
            "count_use_spatial":0,"correct_count_use_spatial":0,
            "count_use_color":0,"correct_count_use_color":0,
            "count_use_shape":0,"correct_count_use_shape":0,
        }

        # 遍历，统计结果
        wrong_line_numbers=[]
        for result_line in result:
            # 首先读取line_number
            line_number=result_line[0] #注意这里读进来是str
            # 如果是空行则跳过
            if result_line[0]=='':
                continue
            # 按照nr3d的说明，如果是人类也没有答对（correct_guess==False)，则跳过
            if (not self.get_correct_guess_info(line_number)) and skip_human_wrong_cases:
                continue
            # 总数记数
            accuracy_count["count_total"]+=1
            # 获取easy信息并给easy的总数记数
            is_easy=self.get_easy_info(line_number)
            easy_setting = 'easy' if is_easy else 'hard'
            accuracy_count['count_%s'%easy_setting]+=1
            # 获取view_dependent信息并记数
            is_view_dep=self.get_view_dep_info(line_number)
            view_dep_setting = 'view_dep' if is_view_dep else 'view_indep'
            accuracy_count['count_%s'%view_dep_setting]+=1
            # 获取left_right信息并记数
            has_left_right=self.get_left_right_info(line_number)
            accuracy_count['count_left_right']+=1 if has_left_right else 0
            # 获取left_right信息并记数
            is_ordinal=self.get_ordinal_info(line_number)
            accuracy_count['count_ordinal']+=1 if is_ordinal else 0
            # use object,spatial,color,shape的记数
            use_lang_settings=['use_object','use_spatial','use_color','use_shape']
            use_lang_settings_used=[]
            for i in range(4):
                setting=use_lang_settings[i]
                if result_line[i+7] in ['True','TRUE','true']:
                    accuracy_count['count_%s'%setting]+=1
                    use_lang_settings_used.append(setting)

            # 给正确案例记数
            if result_line[5]=="True":
                accuracy_count["correct_count_total"]+=1
                accuracy_count['correct_count_%s'%easy_setting]+=1
                accuracy_count['correct_count_%s'%view_dep_setting]+=1 
                accuracy_count['correct_count_left_right']+=1 if has_left_right else 0
                accuracy_count['correct_count_ordinal']+=1 if is_ordinal else 0
                for setting in use_lang_settings_used:
                    accuracy_count['correct_count_%s'%setting]+=1 
            else:
                wrong_line_numbers.append(eval(result_line[0]))
        
        # 打印准确率
        for name in ['total','easy','hard','view_dep','view_indep','left_right','ordinal']+use_lang_settings:
            print(name+" accuracy:")
            correct=accuracy_count["correct_count_"+name]
            total=accuracy_count["count_"+name]
            percentage = -1 if total==0 else correct/total*100
            print("%.2f%% (%d/%d)"%(percentage,correct,total))

    def get_raw_label_2_nyu40_idx(self):
        type2class = {'cabinet':0, 'bed':1, 'chair':2, 'sofa':3, 'table':4, 'door':5,
            'window':6,'bookshelf':7,'picture':8, 'counter':9, 'desk':10, 'curtain':11,
            'refrigerator':12, 'shower curtain':13, 'toilet':14, 'sink':15, 'bathtub':16, 'others':17}  
        scannet_labels = type2class.keys()
        scannet2label = {label: i for i, label in enumerate(scannet_labels)} #从上述18个label映射到idx的字典

        lines = [line.rstrip() for line in open("/share/data/ripl/vincenttann/sr3d/data/scannetv2-labels.combined.tsv")]
        lines = lines[1:]
        raw2label = {}
        for i in range(len(lines)):
            label_classes_set = set(scannet_labels)
            elements = lines[i].split('\t')
            raw_name = elements[1]
            nyu40_name = elements[7]
            if nyu40_name not in label_classes_set:
                raw2label[raw_name] = scannet2label['others']
            else:
                raw2label[raw_name] = scannet2label[nyu40_name] #从scannet中的raw_name映射到上述18个idx之一的字典

        return raw2label

    def get_unique_info(self,scan_id,target_class)->bool:
        # 本函数用于在结果npy文件没有记录scanrefer是否unique的情况下，找到这个信息
        # 读入事先准备好的物体信息，即npy文件
        # 做法参考了scanrefer的代码
        npy_path_train=self.scannet_data_root+"/train/objects_info/objects_info_"+scan_id+".npy"
        npy_path_test=self.scannet_data_root+"/test/objects_info/objects_info_"+scan_id+".npy"
        if os.path.exists(npy_path_train):
            npy_path=npy_path_train
        elif os.path.exists(npy_path_test):
            npy_path=npy_path_test
        else:
            print("object_info.npy file does not exist!!! scan_id:",scan_id)
            return None
        objects_info=np.load(npy_path,allow_pickle=True) #objects_info是gt或3d segmentation得到的场景中所有物体的信息
        obj_idx_in_scene=[]
        for obj in objects_info:
            raw_label=obj['label']
            idx=self.raw_label_2_nyu40_idx[raw_label]
            obj_idx_in_scene.append(idx)
        
        target_class=" ".join(target_class.split("_"))
        target_idx=self.raw_label_2_nyu40_idx[target_class] #将target class映射到18个idx
        is_unique=True if obj_idx_in_scene.count(target_idx)<=1 else False
        return is_unique

    def analyse_result_scanrefer(self,result_path,report_none_gt_error=True):
        # 本函数用于分析scanrefer的结果
        # 首先处理path，如果是列表，就把所有的npy文件对应的np array合并
        if isinstance(result_path,list):
            for idx,path in enumerate(result_path):
                result_single=np.load(path,allow_pickle=True)
                if not idx:
                    result=result_single
                else:
                    result=np.vstack([result,result_single])
        else:
            result=np.load(result_path,allow_pickle=True)
        print("Scanrefer results for:",result_path)

        # 定义记录结果的字典
        accuracy_count={
            "count_total":0,"correct_count_total_25":0,"correct_count_total_50":0,
            "count_unique":0,"correct_count_unique_25":0,"correct_count_unique_50":0,
            "count_multiple":0,"correct_count_multiple_25":0,"correct_count_multiple_50":0,
        }

        # 遍历结果，在accuracy_count中记录相应数据
        iou_list=[]
        correct_answer_exist_count=0
        wrong_line_numbers=[]
        wrong_line_numbers_except=[]
        for result_line in result:
            # 如果是空行则跳过
            if result_line[0]=='':
                continue
            # 读入scan_id和target_class，并获取是否unique
            scan_id=result_line[1]
            target_class=result_line[8]
            if target_class=='toilet_paper_dispense':
                target_class='toilet_paper_dispenser'
            is_unique=self.get_unique_info(scan_id,target_class)
            # 读入iou
            iou=eval(result_line[7])
            # 总数记数
            accuracy_count["count_total"]+=1
            if is_unique:
                accuracy_count["count_unique"]+=1
            else:
                accuracy_count["count_multiple"]+=1
            # iou超过0.25/0.5则给正确数记数
            if iou>=0.5:
                accuracy_count["correct_count_total_50"]+=1
                if is_unique:
                    accuracy_count["correct_count_unique_50"]+=1
                else:
                    accuracy_count["correct_count_multiple_50"]+=1
            if iou>=0.25:
                accuracy_count["correct_count_total_25"]+=1
                if is_unique:
                    accuracy_count["correct_count_unique_25"]+=1
                else:
                    accuracy_count["correct_count_multiple_25"]+=1
            else:
                wrong_line_numbers.append(eval(result_line[0]))
            iou_list.append(iou)
            # 这里还需要记录该案例是否有可能找到正确答案，改为用max_iou比较
            if eval(result_line[10])>=self.scanrefer_iou_thr:
                correct_answer_exist_count+=1
                if iou<=self.scanrefer_iou_thr:
                    wrong_line_numbers_except.append(eval(result_line[0])) #记录有正确答案的情况下的错误案例

        # print("wrong cases line_numbers:",wrong_line_numbers)
        # print("wrong cases line_numbers:",wrong_line_numbers_except)

        # 不同setting和k的Acc@k
        for setting in ['total','multiple','unique']:
            for thr in [50,25]:
                correct=accuracy_count["correct_count_%s_%d"%(setting,thr)]
                total=accuracy_count["count_%s"%setting]
                percentage = -1 if total==0 else correct/total*100
                print("Acc@%.2f %s:"%(thr/100, setting))
                print("%.2f%% (%d/%d)"%(percentage,correct,total))

        # 平均iou
        print("average iou:")
        print("%.3f"%np.average(iou_list))

        # groupfree没提供正确答案的比例
        if report_none_gt_error and not self.use_gt_box:
            total=accuracy_count["count_total"]
            correct=accuracy_count["correct_count_total_50"]
            wrong=total-correct
            no_correct_answer=total-correct_answer_exist_count
            percentage = "-" if wrong==0 else no_correct_answer/wrong*100
            print("Percentage of error caused by 'no correct answer provided by Group Free':")
            print("%.2f%% (%d/%d)"%(percentage,no_correct_answer,wrong))
            # 去除上述情况后的Acc@k
            percentage = "-" if correct_answer_exist_count==0 else correct/correct_answer_exist_count*100
            print("Acc@%.2f without such cases:"%self.scanrefer_iou_thr)
            print("%.2f%% (%d/%d)"%(percentage,correct,correct_answer_exist_count))

    def analyse_result(self,result_path):
        self.load_refer_dataset()
        if self.dataset=='sr3d':
            self.analyse_result_sr3d(result_path)
        elif self.dataset=='nr3d':
            self.analyse_result_nr3d(result_path,skip_human_wrong_cases=True)
        else:
            self.analyse_result_scanrefer(result_path,True)
        return


def find_number_list_in_log(log_file):
    # 打开.log文件进行读取
    with open(log_file, 'r') as file:
        lines = file.readlines()
    # 初始化存储数字的数组
    numbers_a = []
    # 遍历每一行并提取数字a
    for line in lines[1:]:  # 从第二行开始
        parts = line.split()  # 按空格分割
        if len(parts) >= 5 and parts[0] == 'Processing' and parts[2] == 'line':
            try:
                a = int(parts[3].strip(','))  # 提取数字a
                numbers_a.append(a)
            except ValueError:
                pass  # 如果转换失败，跳过该行    
    # 打印提取出的数字数组
    print(numbers_a)
    return numbers_a

def find_number_list_in_failure_log(log_file):
    line_numbers = []

    with open(log_file, 'r') as file:
        lines = file.readlines()
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if line.startswith("LINE NUMBER:"):
                next_line = lines[i + 1].strip()
                if next_line.isdigit():
                    line_numbers.append(int(next_line))
                i += 2
            else:
                i += 1
    return line_numbers

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

#####################
dataset_idx=2  # 0:nr3d, 1:sr3d, 2:scanrefer
mode_idx=0   #<------
#####################



if dataset_idx==0:
    test_config=test_modes_nr3d[mode_idx]
elif dataset_idx==1:
    test_config=test_modes_sr3d[mode_idx]
else:
    test_config=test_modes_scanrefer[mode_idx]

tool=test_config.get('tool') # scanrefer detection tool

print("test config:\n",test_config)

openai_config = {
        'model': test_config['model'],

        'temperature': 0,
        'top_p': 0.0,
        'max_tokens': 'inf',

        'system_message': get_system_message() if test_config['use_code_interpreter'] else '',

        # 'load_path': '',
        'save_path': 'chats',
        'debug': True
    }


"""Nr3d evaluation"""
result_folder_name=test_config['result_folder_name']

refer3d=Refer3d(scannet_data_root="/share/data/ripl/scannet_raw/", 
                script_root="/share/data/ripl/vincenttann/sr3d/",
                dataset=test_config['dataset'], 
                refer_dataset_path=test_config['refer_dataset_path'], 
                result_folder_name=result_folder_name,
                gpt_config=openai_config,
                use_gt_box=test_config['use_gt_box'], 
                ########################################################################################
                # object_filter_result_check_folder_name='eval_results_scanrefer_4_p_gf_valset', #<-------
                ########################################################################################

                # object_filter_result_check_list=['2023-09-15-15-14-28','2023-09-15-18-06-29','2023-09-15-17-13-51','2023-09-15-17-14-30'], #nr3d
                # object_filter_result_check_list=['2023-09-15-23-51-36','2023-09-16-00-02-17','2023-09-16-00-30-09'], #scanrefer
                use_principle=test_config['use_principle'],

                ##############################################
                use_original_viewdep_judge=False,  #<----------
                ##############################################
                scanrefer_tool_name=tool,
                use_priority=test_config['use_priority'],
                use_code_interpreter=test_config['use_code_interpreter']
                )
# refer3d.evaluate_on_GPT(np.arange(2,209))
# refer3d.evaluate_on_GPT(find_number_list_in_failure_log(log_root+"2023-08-15-17-49-04-failure.log")[10:])

###############################################################################
line_number_range=np.arange(0,50)        #<---------------------------------
# line_number_range=np.arange(100,200)+2    #<-----------------------------------
# line_number_range=np.arange(265-2,300)+2    #<-----------------------------------
# refer3d.evaluate_on_GPT(line_number_range, to_compress_prompt=True) #<---------
###############################################################################

"""Nr3d results"""
# refer3d.analyse_result(refer3d.save_path)
# refer3d.analyse_result(log_root+"2023-08-15-11-27-50.npy")
# refer3d.analyse_result(log_root+"2023-08-15-17-15-05.npy")
# refer3d.analyse_result(log_root+"2023-08-15-17-49-04.npy") #72% in 50
# refer3d.analyse_result([log_root+"2023-08-16-15-19-12.npy",])
# formatted_time=["2023-09-14-00-53-10","2023-09-14-00-53-43","2023-09-14-00-54-07"]
# formatted_time="2023-09-15-01-15-42"
# # formatted_time="2023-09-14-00-53-10"

formatted_time=test_config['formatted_time_list']
if isinstance(formatted_time,list):
    print('is list')
    result_path=["%s%s/%s.npy"%(result_folder_name,ft,ft) for ft in formatted_time]
else:
    result_path="%s%s/%s.npy"%(result_folder_name,formatted_time,formatted_time) if formatted_time is not None else None
if result_path:
    refer3d.analyse_result(result_path) 

"""Nr3d correction"""
# formatted_time="2023-09-14-00-53-10"
# line_number_range=np.arange(0,300)+2
# formatted_time="2023-09-14-00-53-43"
# line_number_range=np.arange(300,600)+2
# formatted_time="2023-09-14-00-54-07"
# line_number_range=np.arange(600,1000)+2

# refer3d.self_correction_dataset("/share/data/ripl/vincenttann/sr3d/"+result_folder_name,formatted_time,line_number_range)


"""ScanRefer evaluation"""
# of_result_check_list=["2023-09-11-18-11-35","2023-09-11-18-11-42","2023-09-11-18-12-14","2023-09-11-18-13-12"]
# refer3d=Refer3d(scannet_data_root="/share/data/ripl/scannet_raw/", 
#                 script_root="/share/data/ripl/vincenttann/sr3d/",
#                 dataset="scanrefer", 
#                 refer_dataset_path="/share/data/ripl/vincenttann/sr3d/data/scanrefer/ScanRefer_filtered_train_sampled1000.json",
#                 gpt_config=config,
#                 scanrefer_iou_thr=0.5,
#                 use_gt_box=True,
#                 object_filter_result_check_list=of_result_check_list)
# result_folder_name="eval_results_scanrefer_2stages/"

# # line_number_range=np.arange(84,100)
# # line_number_range=np.arange(135,200)
# # line_number_range=np.arange(235,300)
# # line_number_range=np.arange(312,400)
# # formatted_time=refer3d.evaluate_on_GPT(line_number_range, to_compress_prompt=True, result_folder_name=result_folder_name)

"""ScanRefer results"""
# formatted_time=["2023-09-11-18-11-35","2023-09-11-18-11-42","2023-09-11-18-12-14","2023-09-11-18-13-12"] #使用priority的400个实验，跑通了265个
# # formatted_time=["2023-09-13-01-05-25","2023-09-13-01-05-36","2023-09-13-01-05-49","2023-09-13-01-05-59"]
# formatted_time=["2023-09-13-01-05-25","2023-09-13-01-05-36","2023-09-13-01-05-49","2023-09-13-01-05-59","2023-09-13-14-38-48","2023-09-13-14-38-55","2023-09-13-14-38-59","2023-09-13-14-39-03"] # 不使用priority的400个实验，跑通了383个
# # formatted_time=["2023-09-13-01-05-25","2023-09-13-01-05-36","2023-09-13-01-05-49","2023-09-13-01-05-59","2023-09-13-17-18-48","2023-09-13-17-19-05","2023-09-13-17-19-19","2023-09-13-17-19-33"] 
# if isinstance(formatted_time,list):
#     print('is list')
#     result_path=["%s%s/%s.npy"%(result_folder_name,ft,ft) for ft in formatted_time]
# else:
#     result_path="%s%s/%s.npy"%(result_folder_name,formatted_time,formatted_time)

# refer3d.analyse_result(result_path)
# refer3d.check_scanrefer_answer_exist_percentage(0.5)


"""ScanRefer correction"""
# formatted_time="2023-09-11-18-11-35"
formatted_time=test_config['formatted_time_list']
for ft in formatted_time:
    refer3d.self_correction_dataset(result_folder_path="/share/data/ripl/vincenttann/sr3d/"+test_config['result_folder_name'], formatted_time=ft, line_number_list=np.arange(0,400))



# for sr3d
# refer3d=Refer3d(scannet_data_root="/share/data/ripl/scannet_raw/", 
#                 script_root="/share/data/ripl/vincenttann/sr3d/",
#                 dataset="sr3d", 
#                 refer_dataset_path="/share/data/ripl/vincenttann/sr3d/data/referit3d/sr3d_train_support.csv",
#                 gpt_config=config)

# log_root="/share/data/ripl/vincenttann/sr3d/eval_results/"

# refer3d.evaluate_on_GPT(np.arange(2,330)) 
# refer3d.evaluate_on_GPT(random_sampling(2,3007,'num',50)) #allo
# refer3d.evaluate_on_GPT(random_sampling(2,1191,'num',50)) #support
# refer3d.evaluate_on_GPT(random_sampling(2,2607,'num',50)) #vertical
# refer3d.evaluate_on_GPT(random_sampling(2,53577,'num',50)) #horizontal
# refer3d.evaluate_on_GPT(random_sampling(2,5467,'num',50)) #between

# refer3d.analyse_result(refer3d.save_path)
# refer3d.analyse_result(["./eval_results/2023-08-10-10-39-31.npy","./eval_results/2023-08-10-11-11-55.npy"]) #allo 50
# refer3d.analyse_result("./eval_results/2023-08-10-12-42-46.npy") #support 50
# refer3d.analyse_result("./eval_results/2023-08-10-16-29-30.npy") #vertical 50
# refer3d.analyse_result("./eval_results/2023-08-10-18-54-29.npy") #allocentric 50 有左右向量
# refer3d.analyse_result("./eval_results/2023-08-10-20-03-07.npy") #horizontal
# refer3d.analyse_result("./eval_results/2023-08-11-00-00-16.npy") #between
# refer3d.analyse_result("./eval_results/2023-08-11-12-28-11.npy") #vertical

# refer3d.evaluate_on_GPT(find_number_list_in_log(log_root+"2023-08-11-12-28-11-progress.log")) # vertical 50 same data
# refer3d.analyse_result(log_root+"2023-08-11-16-34-05.npy")

# refer3d.evaluate_on_GPT(find_number_list_in_log(log_root+"2023-08-10-12-42-46-progress.log")) #support 50 same data
# refer3d.analyse_result(log_root+"2023-08-11-17-39-47.npy")

# refer3d.refer_dataset_path="/share/data/ripl/vincenttann/sr3d/data/referit3d/sr3d_train_between.csv"
# refer3d.evaluate_on_GPT(find_number_list_in_log(log_root+"2023-08-11-00-00-16-progress.log")) #between 50 same data
# refer3d.analyse_result(log_root+"2023-08-11-18-41-01.npy")

# refer3d.refer_dataset_path="/share/data/ripl/vincenttann/sr3d/data/referit3d/sr3d_train_allocentric.csv"
# refer3d.evaluate_on_GPT(find_number_list_in_log(log_root+"2023-08-10-18-54-29-progress.log")) #allocentric 50 same data
# # refer3d.analyse_result(log_root+"2023-08-11-19-18-00.npy")
# # refer3d.analyse_result(log_root+"2023-08-15-16-03-15.npy")
# refer3d.analyse_result(log_root+"2023-08-15-16-41-19.npy")

# refer3d.refer_dataset_path="/share/data/ripl/vincenttann/sr3d/data/referit3d/sr3d_train_horizontal.csv"
# refer3d.evaluate_on_GPT(find_number_list_in_log(log_root+"2023-08-10-20-03-07-progress.log")) #horizontal 50 same data
# refer3d.analyse_result(log_root+"2023-08-11-20-02-36.npy")


