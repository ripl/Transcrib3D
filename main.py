# encoding:utf-8
import ast
import csv
import json
import logging
import os
import random
import re
import time
import numpy as np
from copy import deepcopy
from datetime import datetime
from tenacity import RetryError, before_sleep_log, retry, stop_after_attempt, wait_exponential_jitter  # for exponential backoff

from core.code_interpreter import CodeInterpreter
from core.gpt_dialogue import Dialogue
from core.object_filter_gpt4 import ObjectFilter
from core.prompt_text import get_principle, get_principle_sr3d, get_system_message
from config.config import confs_nr3d, confs_scanrefer, confs_sr3d
from utils.utils import *
from utils.read_data import *
from utils.analyse_result import *

# logging.basicConfig(stream=sys.stderr, level=logging.ERROR)
logger = logging.getLogger(__name__ + 'logger')
logger.setLevel(logging.ERROR)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.ERROR)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


class Transcrib3D:
    def __init__(self, workspace_path, scannet_data_root, dataset_type, refer_dataset_path, result_folder_name, gpt_config, scanrefer_iou_thr=0.5, use_gt_box=True, object_filter_result_check_folder_name=None, object_filter_result_check_list=None, use_principle=True, use_original_viewdep_judge=True, use_object_filter=True, scanrefer_tool_name='mask3d', use_priority=False, use_code_interpreter=True, use_camera_position=True, filter_behind_obj=True, obj_info_ablation_type=0) -> None:
        """
        Class initialization.

        Parameters:
            workspace_path (str): Path of Transcrib3D project folder.
            scannet_data_root (str): Path to the ScanNet data folder.
            dataset_type (str): Type of refering dataset. One of [sr3d, nr3d, scanrefer].
            refer_dataset_path (str): Path to the refering dataset file (.csv or .json).
            result_folder_name (str): The name of result folder of a certain experiment setting. It will be under the 'results' folder.
            gpt_config (dict): GPT config dictionary.
            scanrefer_iou_thr (float): The IoU threshold for a ScanRefer case to be judge to correct.
            use_gt_box (bool): To use ground truth bounding boxes of objects in the scene or not.
            object_filter_result_check_folder_name (str): 
            object_filter_result_check_list (list):
            use_principle (bool): To declare some useful logic principles in prompt or not.
            use_original_viewdep_judge (bool): 
            use_object_filter (bool): To use object filter to filter out irrelevant objects or not.
            use_priority (bool): To declare human-designed priorities of constraints (e.g. position, color) or not.
            use_code_interpreter (bool): To use code interpreter during interactive reasoning or not.
            use_camera_position (bool): For scanrefer, to use camera position and pose or not.
            filter_behind_obj (bool): For scanrefer, to filter out objects behind the camera or not.
            obj_info_ablation_type (int): Ablation type of object information.

        Returns:
            None.

        """
        self.workspace_path = workspace_path
        self.scannet_data_root = scannet_data_root
        self.dataset_type = dataset_type
        self.refer_dataset_path = refer_dataset_path
        self.result_folder_name = result_folder_name
        self.gpt_config = gpt_config
        self.scanrefer_iou_thr = scanrefer_iou_thr
        self.use_gt_box = use_gt_box
        self.object_filter_result_check_folder_name = object_filter_result_check_folder_name
        self.object_filter_result_check_list = object_filter_result_check_list
        self.use_principle = use_principle
        self.use_original_viewdep_judge = use_original_viewdep_judge
        self.use_object_filter = use_object_filter
        self.scanrefer_tool_name = scanrefer_tool_name
        self.use_priority = use_priority
        self.use_code_interpreter = use_code_interpreter
        self.obj_info_ablation_type = obj_info_ablation_type
        self.use_camera_position = use_camera_position
        self.filter_behind_obj = filter_behind_obj
        self.token_usage_whole_run = 0
        self.token_usage_this_ques = 0
        self.time_consumed_whole_run = 0
        self.time_consumed_this_ques = 0
        
        self.sr3d_data, self.nr3d_data, self.scanrefer_data = None, None, None

        # self.raw_label_2_nyu40_idx = self.get_raw_label_2_nyu40_idx()


    def filter_out_obj_behind_camera(self, obj_list, camera_info):
        """
        Filter out objects in the half space behind the camera.

        Parameters:
            obj_list (list): List of objects.
            camera_info (dict): A dictionary recording camera position and viewpoint.

        Returns:
            list: List of objects.
        """
        camera_position = camera_info['position']
        camera_lookat = camera_info['lookat']
        lookat_vec = camera_lookat - camera_position
        obj_list_f = []
        for obj in obj_list:
            obj_vec = obj['center_position'] - camera_position
            if np.dot(lookat_vec, obj_vec) >= 0: # the dot product should >= 0
                obj_list_f.append(obj)
        print("Before filter_out_obj_behind_camera: %d objects." % len(obj_list))
        print("After filter_out_obj_behind_camera: %d objects." % len(obj_list_f))
        return obj_list_f

    def non_max_suppression(self, objects_info_f:list, iou_threshold=0.5):
        """
        Filter out overlapped bounding boxes representing the same object. Box with highest confidential score
        will be kept.

        Parameters:
            objects_info_f (list): List of objects(bounding boxes).
            iou_threshold (float): IoU threshold for overlap judgement.

        Returns:
            list: List of objects.
        """
        print("before non_max_suppression: %d objects." % len(objects_info_f))
        # sort in order of conf score
        objects_info_f.sort(key=lambda x: x['score'], reverse=True)

        selected_objects = []

        while len(objects_info_f) > 0:
            current_object = objects_info_f[0]
            selected_objects.append(current_object)
            objects_info_f.pop(0)

            # calculate iou with all other objects in list, delete those has higher iou than threshold.
            objects_info_f = [obj for obj in objects_info_f if calc_iou(current_object['extension'], obj['extension']) < iou_threshold]

        print("after non_max_suppression: %d objects." % len(selected_objects))
        return selected_objects

    @retry(wait=wait_exponential_jitter(initial=20, max=120, jitter=20), stop=stop_after_attempt(5), before_sleep=before_sleep_log(logger, logging.ERROR))  # 20s,40s,80s,120s + random.uniform(0,20)
    def get_gpt_response(self, prompt: str, code_interpreter: CodeInterpreter):
        """
        Get response from GPT(using code interpreter). Using retry from tenacity because the openai token limitation might be reached.
        Measure the token usage and time as well.
        If the reponse does not include "Now the answer is complete", this means the answer is not done. attach an empty user message to continue the conversation.

        Parameters:
            prompt (str): The generated prompt.
            code_interpreter (CodeInterpreter): An instance of CodeInterpreter class.

        Returns:
            list: List of objects.
        """
        # start timing
        call_start_time = time.time()
        # the first call with the original prompt
        response, token_usage_total = code_interpreter.call_llm_with_code_interpreter(prompt)

        # looping until "Now the answer is complete" is in the response, or looping more than 10 times.
        count_response = 0
        while not "Now the answer is complete" in response:
            if count_response >= 10:
                print("Response does not end with 'Now the answer is complete.' !")
                break
            response, token_usage_add = code_interpreter.call_llm_with_code_interpreter('')
            token_usage_total += token_usage_add
            count_response += 1
            print("count_response:", count_response)

        # stop timing and do some statistics
        call_end_time = time.time()
        time_consumed = call_end_time - call_start_time
        self.token_usage_this_ques += token_usage_total
        self.token_usage_whole_run += token_usage_total
        self.time_consumed_this_ques += time_consumed
        self.time_consumed_whole_run += time_consumed

        print("\n*** Refer model: token usage=%d, time consumed=%ds, TPM=%.2f ***" % (token_usage_total, time_consumed, token_usage_total / time_consumed * 60))

        return response

    @retry(wait=wait_exponential_jitter(initial=5, max=30, jitter=5), stop=stop_after_attempt(2), before_sleep=before_sleep_log(logger, logging.ERROR))  # 20s,40s,80s,120s + random.uniform(0,20)
    def get_gpt_response_no_code_interpreter(self, prompt: str, gpt_dialogue: Dialogue):
        """
        Get response from GPT(without code interpreter). Using retry from tenacity because the openai token limitation might be reached.
        Measure the token usage and time as well.
        If the reponse does not include "Now the answer is complete", this means the answer is not done. attach an empty user message to continue the conversation.

        Parameters:
            objects_info_f (list): List of objects(bounding boxes).
            iou_threshold (float): IoU threshold for overlap judgement.

        Returns:
            list: List of objects.
        """
        # start timing
        call_start_time = time.time()
        # the first call with the original prompt
        response, token_usage_total = gpt_dialogue.call_llm(prompt)
        
        # looping until "Now the answer is complete" is in the response, or looping more than 10 times.
        count_response = 0
        while not "Now the answer is complete" in response:
            if count_response >= 10:
                print("Response does not end with 'Now the answer is complete.' !")
                break
            response, token_usage_add = gpt_dialogue.call_llm('')
            token_usage_total += token_usage_add
            # print('Bot:', response)
            count_response += 1
            print("count_response:", count_response)

        # stop timing and do some statistics
        call_end_time = time.time()
        time_consumed = call_end_time - call_start_time
        self.token_usage_this_ques += token_usage_total
        self.token_usage_whole_run += token_usage_total
        self.time_consumed_this_ques += time_consumed
        self.time_consumed_whole_run += time_consumed

        print("\n*** Refer model: token usage=%d, time consumed=%ds, TPM=%.2f ***" % (token_usage_total, time_consumed, token_usage_total / time_consumed * 60))

        return response

    def scanrefer_answer_exist(self, data_index, iou_thr):
        # check whether it's possible to find the correct answer for the given index of scanrefer:
        # if we're using some bounding boxes detected by an object detector(like group-free) or instance segmentor(like mask3d),
        # and if the largest IoU in IoUs of gt box and all boxes detected is less than the threshold(0.25 or 0.5), then it is impossible for the rest of our model to find the correct answer.
        # note the minimum of data_index is 0.
        data = self.scanrefer_data[data_index]
        scan_id = data['scene_id']
        target_id = data['object_id']
        target_class = data['object_name']
        utterance = data['description']
        annotation_id = data['ann_id']
        suffix = '_' + self.scanrefer_tool_name if self.scanrefer_tool_name else ''
        # npy_path_train = os.path.join(self.scannet_data_root, "objects_info%s/objects_info%s_" % (suffix, suffix) + scan_id + ".npy")
        npy_path_train = os.path.join(self.scannet_data_root, "objects_info%s"%suffix, "objects_info%s_%s.npy" % (suffix, scan_id))
        # npy_path_test=self.scannet_data_root+"/test/objects_info%s/objects_info%s_"%(suffix,suffix) +scan_id + ".npy"
        # if os.path.exists(npy_path_train):
        #     npy_path=npy_path_train
        # else:
        #     npy_path=npy_path_test
        # elif os.path.exists(npy_path_test):
        #     npy_path=npy_path_test
        # else:
        #     print("object_info.npy file does not exist!!! scan_id:",scan_id)
        npy_path = npy_path_train
        objects_info = np.load(npy_path, allow_pickle=True)
        gt_box = self.get_scanrefer_gt_box(scan_id, target_id)
        iou_max = 0.0
        iou_max_object = None
        for obj in objects_info:
            box = obj['extension']
            iou = calc_iou(gt_box, box)
            if iou > iou_max:
                iou_max = iou
                iou_max_object = obj
        info = (scan_id, target_id, target_class, utterance, annotation_id, gt_box, iou_max, iou_max_object)
        if iou_max > iou_thr:
            return True, info
        else:
            # print("No box has iou more than %.2f with gt box!!! iou_max is %.3f. Recorded to result and skipped."%(iou_thr,iou_max))
            return False, info

    def check_scanrefer_answer_exist_percentage(self, iou_thr):
        # check all data records in scanrefer and calculate the percentage that answer might exist, given the detected boxes.
        self.scanrefer_data = read_json(self.refer_dataset_path)
        answer_exist_count = 0
        answer_exist_count_unique = 0
        answer_exist_count_multiple = 0
        total_unique = 0
        total_multiple = 0
        total = len(self.scanrefer_data)
        for idx in range(total):
            exist, _ = self.scanrefer_answer_exist(idx, iou_thr)
            data = self.scanrefer_data[idx]
            answer_exist_count += exist
            # 为unique and multiple
            is_unique = self.get_unique_info(data['scene_id'], data['object_name'])
            if is_unique:
                total_unique += 1
                answer_exist_count_unique += exist
            else:
                total_multiple += 1
                answer_exist_count_multiple += exist

        print(self.refer_dataset_path)

        percentage = -1 if total == 0 else answer_exist_count / total * 100
        print("answer exist cases(overall):")
        print("%.2f%% (%d/%d)" % (percentage, answer_exist_count, total))

        percentage = -1 if total_unique == 0 else answer_exist_count_unique / total_unique * 100
        print("answer exist cases(unique):")
        print("%.2f%% (%d/%d)" % (percentage, answer_exist_count_unique, total_unique))

        percentage = -1 if total_multiple == 0 else answer_exist_count_multiple / total_multiple * 100
        print("answer exist cases(multiple):")
        print("%.2f%% (%d/%d)" % (percentage, answer_exist_count_multiple, total_multiple))

    def find_relevant_objects(self, data_index, scan_id, target_id, utterance, npy_path, use_npy_file=True, object_info_list=None, void=False):

        if void:  # not filter, return all objects
            object_filter = ObjectFilter()
            object_filter.load_path = None,
            object_filter.system_message = ''

            all_object_ids = []
            if use_npy_file:
                data = np.load(npy_path, allow_pickle=True)
                for obj in data:
                    if obj['label'] == 'object':
                        continue
                    # line="name=%s,id=%d; "%(obj['label'],obj['id'])
                    # prompt=prompt+line
                    all_object_ids.append(obj['id'])
            else:
                data = object_info_list
                for obj in data:
                    label = obj.get('cls')
                    if label is None:
                        label = obj.get('label')
                    # if obj['cls']=='object':
                    #     continue
                    if label in ['object', 'otherfurniture', 'other', 'others']:
                        continue
                    # line="name=%s,id=%d; "%(label,obj['id'])
                    # prompt=prompt+line
                    all_object_ids.append(obj['id'])

            target_dialogue_path = None
            return all_object_ids, object_filter, target_dialogue_path

        # 新的两步法：先用object filter找到相关物体，在进行refer
        # 如果给出了object_filter_check_list，则在对应文件夹中检查，如果有则直接使用结果
        if self.object_filter_result_check_folder_name is not None:
            target_dialogue_name = "%d_%s_%s_object_filter.json" % (data_index, scan_id, target_id)
            # 定义dialogue文件夹的路径
            folder_paths = ["/share/data/ripl/vincenttann/sr3d/%s/%s/%s_dialogue_jsons/" % (self.object_filter_result_check_folder_name, f_time, f_time) for f_time in self.object_filter_result_check_list]
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
            target_dialogue_path = folder_path + target_dialogue_name
            with open(target_dialogue_path) as f:
                of_response = json.load(f)[-1]['content']
                last_line = of_response.splitlines()[-1]
            object_filter = ObjectFilter()
            relevant_ids = object_filter.extract_all_int_lists_from_text(last_line)
            relevant_dict = object_filter.extract_dict_from_text(last_line)

        else:
            target_dialogue_path = None
            if self.gpt_config['model'] == 'gpt-4-1106-preview':  # 如果refer model用4 turbo，那OF也用
                model = 'gpt-4-1106-preview'
            else:
                model = 'gpt-4'
            print("model used in object filter:", model)
            object_filter = ObjectFilter(model)
            of_start_time = time.time()
            # relevant_ids, token_usage_of = object_filter.filter_objects_by_description(description=utterance, use_npy_file=use_npy_file, objects_info_path=npy_path,object_info_list=object_info_list, to_print=True)
            relevant_dict, token_usage_of = object_filter.filter_objects_by_description(description=utterance, use_npy_file=use_npy_file, objects_info_path=npy_path, object_info_list=object_info_list, to_print=True)
            relevant_ids = []
            for lst in relevant_dict.values():
                relevant_ids += lst

            # 统计时间和token
            of_end_time = time.time()
            time_consumed = of_end_time - of_start_time
            self.token_usage_this_ques += token_usage_of
            self.token_usage_whole_run += token_usage_of
            self.time_consumed_this_ques += time_consumed
            self.time_consumed_whole_run += time_consumed
            print("\n*** Object filter: token usage=%d, time consumed=%ds, TPM=%.2f ***\n" % (token_usage_of, time_consumed, token_usage_of / time_consumed * 60))

        return relevant_ids, relevant_dict, object_filter, target_dialogue_path

    def generate_prompt(self, data_index, to_print=True, deal_with_human_wrong_case=False, deal_with_not_mention_target_class=False):
        """
        Generate prompt for one piece of data record defined by data_index.
        
        Parameters:
            data_index (int): Index of self.sr3d_data/nr3d_data/scanrefer_data. Starts from 0.
            to_print (bool, optional): To print out the generated prompt or not. Defaults to True.
            deal_with_human_wrong_case (bool, optional): For nr3d, whether to deal with cases that human did not answer correctly. Recorded in data['correct_guess']. Defaults to False.
            deal_with_not_mention_target_class (bool, optional): For nr3d, whether to deal with cases that the utterance does not contain the object class name. Recorded in data['mentions_target_class']. Defaults to False.

        Returns:
            str: The generated prompt.
            tuple: A tuple of some information of this case.
            list: A list of IDs of relevant objects. Only used for nr3d and scanrefer.
        """
        # read in data with the given data_index
        if self.dataset_type == 'sr3d':
            data = self.sr3d_data[data_index]
        elif self.dataset_type == 'nr3d':
            data = self.nr3d_data[data_index]
        else:
            data = self.scanrefer_data[data_index]

        # directly return if certain conditions are met.
        if (self.dataset_type == 'sr3d' or self.dataset_type == 'nr3d') and (not deal_with_human_wrong_case) and (data['correct_guess'] in ['False', 'FALSE', 'false']):
            return -1, -1, -1
        if (self.dataset_type == 'sr3d' or self.dataset_type == 'nr3d') and (not deal_with_not_mention_target_class) and (data['mentions_target_class'] in ['False', 'FALSE', 'false']):
            return -2, -2, -2
        
        # read in scan_id
        scan_id = data['scene_id'] if self.dataset_type == 'scanrefer' else data['scan_id']
        if to_print:
            print("scan_id:", scan_id)

        # read in refered class and object ids
        target_class = data['object_name'] if self.dataset_type == 'scanrefer' else data["instance_type"]
        target_id = data['object_id'] if self.dataset_type == 'scanrefer' else data["target_id"]

        # read in utterance
        utterance = data['description'] if self.dataset_type == 'scanrefer' else data["utterance"]
        if not utterance.endswith("."):
            utterance += "."

        # read in reference type, distractors_ids, achor_types and anchor_ids of sr3d
        if self.dataset_type == 'sr3d':
            reference_type = data["coarse_reference_type"]
            distractor_ids = eval(data["distractor_ids"])
            anchor_classes = data["anchors_types"]
            anchor_ids = eval(data["anchor_ids"])

        # read in some information of nr3d
        elif self.dataset_type == 'nr3d':
            mentions_target_class, uses_object_lang, uses_spatial_lang, uses_color_lang, uses_shape_lang = data["mentions_target_class"], data["uses_object_lang"], data["uses_spatial_lang"], data["uses_color_lang"], data["uses_shape_lang"]

        # read in some information of scanrefer, including the camera information
        else:
            annotation_id = data['ann_id']
            camera_info_aligned = get_scanrefer_camera_info_aligned(os.path.join(self.workspace_path, "data"), scan_id, target_id, annotation_id)

        # read in the prepared object information (.npy file)
        npy_path_train = os.path.join(self.scannet_data_root, "objects_info_%s" % self.scanrefer_tool_name, "objects_info_%s_%s.npy" % (self.scanrefer_tool_name, scan_id)) if (self.dataset_type == 'scanrefer' and not self.use_gt_box) else os.path.join(self.scannet_data_root, "objects_info", "objects_info_%s.npy" % scan_id)
        # npy_path_test=self.scannet_data_root+"/test/objects_info_%s/objects_info_%s_"%(self.scanrefer_tool_name,self.scanrefer_tool_name) +scan_id + ".npy" if (self.dataset_type=='scanrefer' and not self.use_gt_box) else self.scannet_data_root+"/test/objects_info/objects_info_"+scan_id+".npy"
        # if os.path.exists(npy_path_train):
        #     npy_path=npy_path_train
        # elif os.path.exists(npy_path_test):
        #     npy_path=npy_path_test
        # else:
        #     print("object_info.npy file does not exist!!! scan_id:",scan_id)
        #     return None, None, None
        npy_path = npy_path_train
        objects_info = np.load(npy_path, allow_pickle=True)  # objects_info是gt或3d segmentation得到的场景中所有物体的信息

        # For scanrefer, filter out the objects in the half space behind the camera.
        if self.dataset_type == 'scanrefer' and self.use_camera_position and self.filter_behind_obj:
            objects_info = self.filter_out_obj_behind_camera(objects_info, camera_info_aligned)

        # For scanrefer, if ground truth boxes are not used, we have to filter out boxes with low confidential scores and conduct non-max suppression
        if self.dataset_type == 'scanrefer' and not self.use_gt_box:
            objects_info_f = []
            for obj in objects_info:
                score = obj.get('score')
                if score is None or score > 0.4:
                    objects_info_f.append(obj)
                    if score is None:
                        print("get confidential score failed!!")
            if score is not None:
                objects_info = self.non_max_suppression(objects_info_f)

        # 统计场景中所有物体的类别，用于scanrefer的unique/multiple分类
        # obj_idx_in_scene=[]
        # for obj in objects_info:
        #     obj_idx_in_scene.append(self.raw_label_2_nyu40_idx[obj['label']])
        # target_idx=self.raw_label_2_nyu40_idx[' '.join(target_class.split('_'))]
        # is_unique=True if obj_idx_in_scene.count(target_idx)<=1 else False
        is_unique = True

        # For sr3d, relevant objects include the target, the distractors, and the anchors.
        if self.dataset_type == 'sr3d':
            objects_related = []
            anchor_has_front = True
            objects_related.append(objects_info[int(target_id)])
            for id in distractor_ids:
                objects_related.append(objects_info[int(id)])
            for id in anchor_ids:
                objects_related.append(objects_info[int(id)])
                anchor_has_front = anchor_has_front and objects_info[int(id)]['has_front']
            # TODO: sort objects by id?
            
        # For nr3d and scanrefer, use object filter to find relevant objects.
        else:
            relevant_ids, relevant_dict, object_filter, target_dialogue_path = self.find_relevant_objects(data_index, scan_id, target_id, utterance, npy_path, use_npy_file=False, object_info_list=objects_info)
            # create a mapping from id to the relevant obj name in description
            id_to_name_in_description = {}
            for name, ids in relevant_dict.items():
                for id in ids:
                    id_to_name_in_description[id] = name
            objects_related = objects_info if (relevant_ids is None) else [obj for obj in objects_info if obj['id'] in relevant_ids]

        # # 对于sr3d记录anchor_has_front
        # if self.dataset_type=='sr3d':
        #     anchor_has_front=True
        #     for id in anchor_ids:
        #         anchor_has_front=anchor_has_front and objects_info[int(id)]['has_front']

        # get the center of the scene
        scene_center = get_scene_center(objects_info)

        # Generate the background part of the prompt
        prompt = scan_id + ":objs with quant description based on r-h Cartesian coord sys with x-y-z axes, x-y plane=ground, z-axis=up/down. coords format [x, y, z].\n"
        if self.dataset_type == 'nr3d':
            prompt = prompt + "Scene center:%s. If no direction vector, observer at center for obj orientation.\n" % remove_spaces(str(scene_center))
        elif self.dataset_type == 'scanrefer':
            if self.use_camera_position:
                prompt = prompt + "Scene center:%s.\n" % remove_spaces(str(scene_center))
                prompt = prompt + "Observer position:%s.\n" % remove_spaces(str(round_list(camera_info_aligned['position'], 2)))   
            else:
                prompt = prompt + "Scene center:%s. If no direction vector, observer at center for obj orientation.\n" % remove_spaces(str(scene_center))

        #Iterate through relevant objects and generate quantatitive description.
        prompt = prompt + "objs list:\n"
        lines = []
        for obj in objects_related:
            # position
            center_position = obj['center_position']
            center_position = round_list(center_position, 2)
            # size
            size = obj['size']
            size = round_list(size, 2)
            # extension
            extension = obj['extension']
            extension = round_list(extension, 2)
            # direction vector. only used for some objects in sr3d and nr3d
            if obj['has_front'] and self.dataset_type != 'scanrefer':
                # front vector
                front_point = np.array(obj['front_point'])
                center = np.array(obj['obb'][0:3])
                direction_vector = front_point - center
                direction_vector_normalized = direction_vector / np.linalg.norm(direction_vector)
                # left and right vector
                front_vector = round_list(direction_vector_normalized, 2)
                up_vector = np.array([0, 0, 1])
                left_vector = round_list(np.cross(direction_vector_normalized, up_vector), 2) # the left side when observer faces the front of the object
                right_vector = round_list(np.cross(up_vector, direction_vector_normalized), 2)
                behind_vector = round_list(-np.array(front_vector), 2)
                direction_info = ";direction vectors:front=%s,left=%s,right=%s,behind=%s\n" % (front_vector, left_vector, right_vector, behind_vector)
            else:
                direction_info = "\n"

            # For sr3d, describe center and size.
            if self.dataset_type == 'sr3d':
                # sr3d ablation study
                if self.obj_info_ablation_type == 1:
                    # no size
                    line = f'{obj["label"]},id={obj["id"]},ctr={remove_spaces(str(center_position))}'
                elif self.obj_info_ablation_type == 2:
                    # min+max
                    line = f'{obj["label"]},id={obj["id"]},xmin={np.round(center_position[0]-size[0]/2, 2)},xmax={np.round(center_position[0]+size[0]/2, 2)},ymin={np.round(center_position[1]-size[1]/2, 2)},ymax={np.round(center_position[1]+size[1]/2, 2)},zmin={np.round(center_position[2]-size[2]/2, 2)},zmax={np.round(center_position[2]+size[2]/2, 2)}'
                elif self.obj_info_ablation_type == 3:
                    # reversed
                    line = f'size={remove_spaces(str(size))},ctr={remove_spaces(str(center_position))},id={obj["id"]},{obj["label"]}'
                else:
                    # vanilla
                    line = f'{obj["label"]},id={obj["id"]},ctr={remove_spaces(str(center_position))},size={remove_spaces(str(size))}'

            # For nr3d and scanrefer, describe center, size and color.
            else:
                rgb = obj['median_rgba'][0:3] if (self.dataset_type == 'scanrefer' and not self.use_gt_box) else obj['avg_rgba'][0:3]
                hsl = round_list(rgb_to_hsl(rgb), 2)

                # line="%s,id=%s,ctr=%s,size=%s,RGB=%s" %(obj['label'], obj['id'], remove_spaces(str(center_position)), remove_spaces(str(size)), remove_spaces(str(rgb) )) #rgb
                # line="%s,id=%s,ctr=%s,size=%s,HSL=%s" %(obj['label'], obj['id'], remove_spaces(str(center_position)), remove_spaces(str(size)), remove_spaces(str(hsl))) #hsl
                line = "%s(relevant to %s),id=%s,ctr=%s,size=%s,HSL=%s" % (obj['label'], id_to_name_in_description[obj['id']], obj['id'], remove_spaces(str(center_position)), remove_spaces(str(size)), remove_spaces(str(hsl)))
            
            # Append direction info to line and append it to lines
            lines.append(line + direction_info)
        
        # ablation study 4: shuffle the lines
        if self.obj_info_ablation_type == 4:
            random.seed(0)
            random.shuffle(lines)
            
        # append lines to prompt
        prompt += ''.join(lines)
        
        # the instruction part of the prompt
        line = "Instruction:find the one described object in description: \n\"%s\"\n" % utterance
        prompt = prompt + line
        # if self.dataset_type=='sr3d':
        #     prompt=prompt+get_principle_sr3d(utterance) if self.use_principle else prompt
        # else:
        #     prompt=prompt+get_principle(utterance,self.use_priority) if self.use_principle else prompt
        # if not self.dataset_type=='sr3d':
        #     # prompt=prompt+" Howerver, if the direction vector of A is not provided, you should use other information to identify the referred object instead of assuming a direction vector."

        # some additional prompt engineering
        prompt = prompt + "\nThere is exactly one answer, so if you receive multiple answers, consider other constraints; if get no answers, loosen constraints."
        prompt = prompt + "\nWork this out step by step to ensure right answer."
        prompt = prompt + "\nIf the answer is complete, add \"Now the answer is complete -- {'ID':id}\" to the end of your answer(that is, your completion, not your code), where id is the id of the referred obj. Do not add anything after."

        if to_print:
            print("--------------------------------------------")
            print("Generated prompt:\n" + prompt)
            print("--------------------------------------------")
            print("Right answer:", target_id)
            print("")
            
        # some inforation to be returned
        if self.dataset_type == 'sr3d':
            relevant_ids = None
            info = (scan_id, target_id, target_class, distractor_ids, reference_type, utterance, anchor_has_front)
        elif self.dataset_type == 'nr3d':
            info = (scan_id, target_id, target_class, utterance, mentions_target_class, uses_object_lang, uses_spatial_lang, uses_color_lang, uses_shape_lang, object_filter, target_dialogue_path)
        else:
            gt_box = get_scanrefer_gt_box(scan_id, target_id)
            info = (scan_id, target_id, target_class, utterance, annotation_id, objects_related, gt_box, object_filter, target_dialogue_path, is_unique)

        return prompt, info, relevant_ids

    def extract_answer_id_from_last_line(self, last_line, random_choice_list=[0,]):
        # 如果没有按照预期格式回复则随机选取(Sr3d)或直接选成0(Nr3d和Scanrefer);按预期格式恢复则提取答案
        wrong_return_format = False
        last_line_split = last_line.split('--')
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
                answer_id = extracted_dict['ID']
                # 如果确实以 Now the answer is complete -- {'ID': xxx} 的格式回复了，但是xxx不是数字（例如是None），也能随机选。
                if not isinstance(answer_id, int):
                    if isinstance(answer_id, list) and all([isinstance(e, int) for e in answer_id]):
                        print("Wrong answer format: %s. random choice from this list" % str(answer_id))
                        answer_id = random.choice(answer_id)
                    else:
                        print("Wrong answer format: %s. No dict found. Random choice from relevant objects." % str(answer_id))
                        answer_id = random.choice(random_choice_list)
                    wrong_return_format = True
            except BaseException:
                print("Wrong answer format!! No dict found. Random choice.")
                answer_id = random.choice(random_choice_list)
                wrong_return_format = True
        else:
            print("Wrong answer format!! No dict found. Random choice.")
            answer_id = random.choice(random_choice_list)
            wrong_return_format = True

        return answer_id, wrong_return_format

    def evaluate_on_GPT(self, line_numbers):
        """
        @descr  the most important function. run evluation for the given data records decided by the line_numbers.  then save the result table to npy file.
        @param  line_numbers: a list of data record indices. for sr3d and nr3d, the minimum is 2. for scanrefer, it's 0.
        """
        # first load the refering dataset.
        load_refer_dataset(self, line_numbers)

        # create a table for recording results. format:
        #       0     #     1   #       2        #     3     #     4     #      5     #          6            #         7
        # sr3d:
        # line_number # scan_id # reference_type # target_id # answer_id # is_correct #  anchors_has_front    #
        # nr3d:
        # line_number # scan_id #    None        # target_id # answer_id # is_correct # mentions_target_class # uses_object_lang # uses_spatial_lang # uses_color_lang # uses_shape_lang
        # scanrefer:
        #  dscrp_num  # scan_id #    ann_id      # target_id # answer_id #   gt_box   #     answer_box        #     iou          #  object_class      # correct_answer_exist # iou_max   # is_unique

        dataset_len = len(line_numbers)
        results_table = np.zeros([dataset_len, 12], dtype='<U21')

        # record current time for the name of the files.
        current_time = datetime.now()
        formatted_time = current_time.strftime("%Y-%m-%d-%H-%M-%S")
        print("formatted_time:", formatted_time)

        # create a result folder for the chosen test mode if it does not exist.
        result_folder = os.path.join(self.workspace_path, 'results', self.result_folder_name)
        if not os.path.exists(result_folder):
            os.makedirs(result_folder)

        # the subfolder of the current experiment. named after the time.
        # results_sub_folder = self.workspace_path + self.result_folder_name + formatted_time + '/'
        results_sub_folder = os.path.join(result_folder, formatted_time)
        if not os.path.exists(results_sub_folder):
            os.makedirs(results_sub_folder)

        # path of relevant files.
        process_log_file = os.path.join(results_sub_folder, "%s-progress.log" % formatted_time)
        success_log_file = os.path.join(results_sub_folder, "%s-success.log" % formatted_time)
        failure_log_file = os.path.join(results_sub_folder, "%s-failure.log" % formatted_time)
        result_npy_file =  os.path.join(results_sub_folder, "%s.npy" % formatted_time)
        dialogue_json_folder = os.path.join(results_sub_folder, "%s_dialogue_jsons" % formatted_time)
        os.makedirs(dialogue_json_folder)

        # iterate through the chosen part of dataset
        for idx, line_number in enumerate(line_numbers):
            # print and record the process
            print("\n\nProcessing %s line %d, %d/%d." % (self.dataset_type, line_number, idx + 1, dataset_len))

            with open(process_log_file, 'a') as f:
                if idx == 0:
                    f.write(self.refer_dataset_path + '\n')
                    f.write(str(list(line_numbers)) + '\n')
                f.write("\nProcessing %s line %d, %d/%d. " % (self.dataset_type, line_number, idx + 1, dataset_len))

            # for scanrefer, check if answer might exist. if not, record this and save.
            if self.dataset_type == 'scanrefer':
                exist, info = self.scanrefer_answer_exist(line_number, iou_thr=0.25)
                scan_id, target_id, target_class, utterance, annotation_id, gt_box, iou_max, iou_max_object = info
                results_table[idx][9] = exist  # correct_answer_exist
                results_table[idx][10] = iou_max  # iou_max
                with open(process_log_file, 'a') as f:
                    f.write("iou_max=%.3f. " % iou_max)
                if not exist and not self.use_gt_box:
                    results_table[idx][0] = line_number
                    results_table[idx][1] = scan_id
                    results_table[idx][2] = annotation_id
                    results_table[idx][3] = target_id
                    results_table[idx][5] = str(list(gt_box))
                    results_table[idx][6] = None  # answer box
                    results_table[idx][7] = 0.0  # iou
                    results_table[idx][8] = target_class
                    results_table[idx][10] = iou_max
                    # results_table[idx][11]=is_unique
                    with open(process_log_file, 'a') as f:
                        f.write("No correct answer, iou_max is %.3f, under 0.25, Skipped." % iou_max)
                    np.save(result_npy_file, results_table)
                    print("results saved to: %s\n\n" % result_npy_file)
                    continue

            # token and time usage
            self.time_consumed_this_ques = 0
            self.token_usage_this_ques = 0

            # generate prompt
            prompt, info, relevant_ids = self.generate_prompt(line_number, to_print=True)
            if prompt is None:
                with open(process_log_file, 'a') as f:
                    f.write("prompt not generated. Perhaps the object_info npy file does not exist.")
                continue

            elif prompt == -1:
                with open(process_log_file, 'a') as f:
                    # f.write("Human failed to find this object: '%s'. Skipped." % utterance)
                    f.write("Human failed to find object, line number '%s'. Skipped." % line_number)
                continue

            elif prompt == -2:
                with open(process_log_file, 'a') as f:
                    f.write("Not mention target class, line number: '%s'. Skipped." % line_number)
                continue
                
            # read some information from info
            if self.dataset_type == 'sr3d':
                scan_id, target_id, target_class, distractor_ids, reference_type, utterance, anchor_has_front = info
                object_filter = ObjectFilter()
                prev_of_dialogue_path = None
            elif self.dataset_type == 'nr3d':
                scan_id, target_id, target_class, utterance, mentions_target_class, uses_object_lang, uses_spatial_lang, uses_color_lang, uses_shape_lang, object_filter, prev_of_dialogue_path = info
            else:
                scan_id, target_id, target_class, utterance, annotation_id, objects_related, gt_box, object_filter, prev_of_dialogue_path, is_unique = info
            object_filter: ObjectFilter

            # 尝试获取GPT回复。如果出现Retry Error，那就last_line随便设置，最终导致wrong_format=True，随机选取
            get_gpt_response_success = True
            try:
                if self.use_code_interpreter:
                    code_interpreter = CodeInterpreter(**self.gpt_config)
                    response = self.get_gpt_response(prompt, code_interpreter)
                else:
                    gpt_dialogue = Dialogue(**self.gpt_config)
                    response = self.get_gpt_response_no_code_interpreter(prompt, gpt_dialogue)
                    code_interpreter = gpt_dialogue  # 这里必须给code_interpreter绑定一个值
                print("\n*** This question: token usage=%d, time consumed=%ss, TPM=%.2f ***" % (self.token_usage_this_ques, self.time_consumed_this_ques, self.token_usage_this_ques / self.time_consumed_this_ques * 60))
                print("*** Whole run: token usage=%d, time consumed=%ss, TPM=%.2f ***\n" % (self.token_usage_whole_run, self.time_consumed_whole_run, self.token_usage_whole_run / self.time_consumed_whole_run * 60))
            except RetryError as r:
                print(r)
                with open(process_log_file, 'a') as f:
                    f.write("ReTry Error.")
                response = "Fail to get response from GPT. RetryError in func get_gpt_response"
                last_line = "Nonesense"
                get_gpt_response_success = False
                code_interpreter = Dialogue(**self.gpt_config)  # 这里必须给code_interpreter绑定一个值

            # 处理GPT的回复 （如果成功获取）
            if get_gpt_response_success:
                print("--------------------------------------------")
                print("DIALOGUE:")
                code_interpreter.print_pretext()
                print("--------------------------------------------")
                last_line = response.splitlines()[-1] if len(response) > 0 else ''
                print(type(last_line))
                print("last_line:", last_line)

            # 从last_line中获取answer_id，如果格式不符合要求则从relevant_ids中随机选取
            random_choice_list = np.append(distractor_ids, target_id) if self.dataset_type == 'sr3d' else relevant_ids
            answer_id, wrong_return_format = self.extract_answer_id_from_last_line(last_line, random_choice_list)

            # 对于scanrefer，要找到answer_id对应的box并计算iou
            if self.dataset_type == 'scanrefer':
                for obj in objects_related:
                    if obj['id'] == answer_id:
                        answer_object = obj
                        break
                # answer_object=objects_related[answer_id]
                answer_box = center_size_to_extension(np.append(answer_object['center_position'], answer_object['size']))
                iou = calc_iou(answer_box, gt_box)

            # 在表格中记录相关信息
            results_table[idx][0] = line_number
            results_table[idx][1] = scan_id
            results_table[idx][3] = target_id
            results_table[idx][4] = answer_id
            if self.dataset_type == 'sr3d':
                results_table[idx][2] = reference_type
                results_table[idx][6] = anchor_has_front
            elif self.dataset_type == 'nr3d':
                results_table[idx][2] = 'None'
                results_table[idx][6] = mentions_target_class
                results_table[idx][7] = uses_object_lang
                results_table[idx][8] = uses_spatial_lang
                results_table[idx][9] = uses_color_lang
                results_table[idx][10] = uses_shape_lang
            else:
                results_table[idx][2] = annotation_id
                results_table[idx][5] = str(list(gt_box))
                results_table[idx][6] = str(list(answer_box))
                results_table[idx][7] = iou
                results_table[idx][8] = target_class
                # results_table[idx][10]=is_unique

            # update 'printed_pretext' for code_interpreter and object_filter
            code_interpreter.print_pretext(to_print_out=False)
            object_filter.print_pretext(to_print_out=False)

            # 对于sr3d和nr3d，比较answer_id和target_id来判断是否回答正确
            if self.dataset_type == 'sr3d' or self.dataset_type == 'nr3d':
                if str(answer_id) == str(target_id):
                    answer_correct = True
                    print("answer correct.")
                    results_table[idx][5] = True
                    # 记录正确信息
                    self.log_info(line_number, scan_id, utterance, object_filter.printed_pretext, code_interpreter.printed_pretext, success_log_file, target_id, answer_id)
                    with open(process_log_file, 'a') as f:
                        f.write("answer correct.")
                    # 如果是错误返回格式，随后蒙对的，也要记录在错误log中
                    if wrong_return_format:
                        self.log_info(line_number, scan_id, utterance, object_filter.printed_pretext, code_interpreter.printed_pretext, failure_log_file, target_id, answer_id)
                        with open(process_log_file, 'a') as f:
                            f.write("But it's a guess after receiving wrong format.")
                else:
                    answer_correct = False
                    print("answer wrong!")
                    results_table[idx][5] = str(False)
                    print("Error info:\nutterance: %s\ntarget_id:%s\nanswer_id:%s\nGPT last response:%s" % (utterance, str(target_id), str(answer_id), response))
                    # 记录错误信息
                    self.log_info(line_number, scan_id, utterance, object_filter.printed_pretext, code_interpreter.printed_pretext, failure_log_file, target_id, answer_id)
                    with open(process_log_file, 'a') as f:
                        f.write("answer wrong!")

            # 对于scanrefer，按iou是否超过阈值来判断
            else:
                target_id_text = str(target_id) + "(ScanNet) / " + str(iou_max_object['id']) + "(%s)" % self.scanrefer_tool_name
                if iou > self.scanrefer_iou_thr:
                    answer_correct = True
                    print("answer correct: IoU=%.3f" % iou)
                    # 记录正确信息
                    self.log_info(line_number, scan_id, utterance, object_filter.printed_pretext, code_interpreter.printed_pretext, success_log_file, target_id_text, answer_id, iou, iou_max)
                    with open(process_log_file, 'a') as f:
                        f.write("answer correct. iou=%.3f" % iou)
                else:
                    answer_correct = False
                    print("answer wrong! IoU=%.3f" % iou)
                    # 记录错误信息
                    self.log_info(line_number, scan_id, utterance, object_filter.printed_pretext, code_interpreter.printed_pretext, failure_log_file, target_id_text, answer_id, iou, iou_max)
                    with open(process_log_file, 'a') as f:
                        f.write("answer wrong! iou=%.3f" % iou)

            # 保存对话到json文件
            if prev_of_dialogue_path:
                import shutil
                shutil.copy(prev_of_dialogue_path, dialogue_json_folder)
                print("copy previous object filter dialogue %s to %s" % (prev_of_dialogue_path, dialogue_json_folder))
            else:
                object_filter_json_name = "%d_%s_%s_object_filter.json" % (line_number, scan_id, target_id)
                object_filter.save_pretext(dialogue_json_folder, object_filter_json_name)
            success_text = "success" if (answer_correct and not wrong_return_format) else "failure"
            refer_json_name = "%d_%s_%s_refer_%s.json" % (line_number, scan_id, target_id, success_text)
            code_interpreter.save_pretext(dialogue_json_folder, refer_json_name)

            # 保存结果表格
            np.save(result_npy_file, results_table)
            print("results saved to: %s\n\n" % result_npy_file)

        self.save_path = result_npy_file

        return formatted_time

    def self_correction(self, failure_diagolue_path, target_id, target_class):
        # 读入failure dialogue备用
        with open(failure_diagolue_path, 'r') as f:
            failure_dialogue = json.load(f)
            failure_dialogue_length = len(failure_dialogue)
            # original_user_dialogue=failure_dialogue[0:2] # system and user

        # 初始化code interpreter
        gpt_config = deepcopy(self.gpt_config)
        gpt_config['load_path'] = failure_diagolue_path
        code_interpreter = CodeInterpreter(**gpt_config)
        code_interpreter.print_pretext()

        # 准备prompt并让gpt自行发现问题，直到其输出Now the answer has complete
        print("\nself correcting...\n")
        correction_prompt = "The correct answer is %s %d. Can you double check the information of %s %d and the given prompt and see where you got wrong? Still, add \"Now the answer is complete -- {'ID':id}\" to the end of your answer, where id is the correct id of the referred obj." % (target_class, int(target_id), target_class, int(target_id))
        print("correctin prompt:", correction_prompt)
        self.get_gpt_response(correction_prompt, code_interpreter)
        print("--------------------------------------------")
        print("ORIGINAL PROMPT AND CORRECTION DIALOGUE:")
        code_interpreter.print_pretext(print_system_and_user_first_prompt=False)
        print("--------------------------------------------")
        self_correction_length = len(code_interpreter.pretext) - failure_dialogue_length  # self correction新增的长度

        # 删除gpt之前的错误推理，并让其完整输出推理过程
        print("\nregenerating reasoning process...\n")
        del code_interpreter.pretext[2:failure_dialogue_length]
        regenerate_prompt = "Now you have the correct reasoning and result. Can you generate the whole reasoning process to get this correct answer from the very beginning? Do not mention that you know the correct answer. You cannot use the code execution result above and have to generate code when needed.  When answer step by step, stop whenever you feel there is need to generate python code and wait for the result from the code execution. Remember to use print() function to print out the result and keep two decimal places for numbers."
        print("regenerate prompt:", regenerate_prompt)
        response = self.get_gpt_response(regenerate_prompt, code_interpreter)
        print("--------------------------------------------")
        print("RE-GENERATED REASONING DIALOGUE:")
        code_interpreter.print_pretext(print_system_and_user_first_prompt=False)
        print("--------------------------------------------")

        # 提取结果并检查是否为正确答案
        last_line = response.splitlines()[-1] if len(response) > 0 else ''
        answer_id, _ = self.extract_answer_id_from_last_line(last_line)
        if str(answer_id) == str(target_id):
            # correction后答案正确，删除correction prompt部分，只保留original prompt和推理过程
            del code_interpreter.pretext[2:2 + self_correction_length]
            correction_success = True
        else:
            print("wrong answer id after correction!!")
            correction_success = False

        return code_interpreter, correction_success

    def self_correction_dataset(self, result_folder_path, formatted_time, line_number_list):
        # 首先确定refer数据集
        refer_dataset = load_refer_dataset(self)

        # 定义dialogue文件夹路径
        # dialogue_folder_path = "%s%s/%s_dialogue_jsons/" % (result_folder_path, formatted_time, formatted_time)
        dialogue_folder_path = os.path.join(result_folder_path, formatted_time, "%s_dialogue_jsons"%formatted_time)

        # 遍历指定line_number_list
        for line_number in line_number_list:
            # 获取相关数据
            data_line = refer_dataset[line_number]
            scan_id = data_line['scene_id'] if self.dataset_type == 'scanrefer' else data_line['scan_id']
            target_id = data_line['object_id'] if self.dataset_type == 'scanrefer' else data_line['target_id']
            target_class = data_line['object_name'] if self.dataset_type == 'scanrefer' else data_line['instance_type']
            # 定义原始failure dialogue的路径
            dialogue_path = os.path.join(dialogue_folder_path, "%d_%s_%s_refer_failure.json" % (line_number, scan_id, target_id) )
            # correction dialogue的路径
            correction_dialogue_name = "%d_%s_%s_refer_correction.json" % (line_number, scan_id, target_id)
            # correction_dialogue_path = dialogue_folder_path + correction_dialogue_name
            correction_dialogue_path = os.path.join(dialogue_folder_path, correction_dialogue_name)
            # 检查correction dialogue是否存在，如果已经存在则跳过
            if os.path.exists(correction_dialogue_path):
                print("correction dialogue %s already exists! skipped." % correction_dialogue_path)
                continue
            # 如果failure dialogue存在（说明是错误案例），则改正后保存到新文件
            if os.path.exists(dialogue_path):
                print("failure dialogue found: " + dialogue_path)
                try:
                    code_interpreter, correction_success = self.self_correction(dialogue_path, target_id, target_class)
                except Exception as e:
                    print("exception arised!!!")
                    print(e)
                    code_interpreter = CodeInterpreter()
                    correction_success = False

                if correction_success:
                    code_interpreter.save_pretext(dialogue_folder_path, correction_dialogue_name)
                    print("correction succeed! saved to: %s%s" % (dialogue_folder_path, correction_dialogue_name))
                else:
                    correction_dialogue_name = "%d_%s_%s_refer_correction_fail.json" % (line_number, scan_id, target_id)
                    code_interpreter.save_pretext(dialogue_folder_path, correction_dialogue_name)
                    print("correction fail! saved to: %s%s" % (dialogue_folder_path, correction_dialogue_name))
            # 如果不存在（说明是正确案例或line_number不正确），则跳过
            else:
                print("failure dialogue not found! " + dialogue_path)

    def log_info(self, line_number, scan_id, utterance, dialogue_object_filter, dialogue_refer, log_file_path, correct_id, answer_id, iou=None, max_iou=None):
        info = "------------------------------------------------------------\n"
        info = info + "LINE NUMBER: \n" + str(line_number) + "\n\n"
        info = info + "SCAN ID: \n" + scan_id + "\n\n"
        info = info + "UTTERANCE: \n" + utterance + "\n\n"
        info = info + "CORRECT ID: \n" + str(correct_id) + "\n\n"
        info = info + "ANSWER ID: \n" + str(answer_id) + "\n\n"
        if not (iou is None):
            info = info + "IoU:\n%.3f\n\n" % iou
            info = info + "MAX IoU:\n%.3f\n\n" % max_iou
        info = info + "DIALOGUE OBJECT FILTER: \n" + dialogue_object_filter + "\n"
        info = info + "DIALOGUE REFER: \n" + dialogue_refer + "\n" + \
            "------------------------------------------------------------\n\n\n"

        with open(log_file_path, 'a') as f:
            f.write(info)

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Transcrib3D")

    parser.add_argument("--scannet_data_root", type=str, help="Path of folder that contains scannet scene folders such as 'scene0000_00'.")
    parser.add_argument("--workspace_path", type=str, help="Path of the Transcribe3D project folder.")
    parser.add_argument("--mode", type=str, choices=["eval", "result", "self_correct", "check_scanrefer"], help="Mode of operation     (eval or result)")
    parser.add_argument("--dataset_type", type=str, choices=["nr3d", "sr3d", "scanrefer"], help="Choose the refering dataset.")
    parser.add_argument("--conf_idx", type=int, help="Configuration index in file config.py.")
    parser.add_argument("--range", type=int, nargs='*', help="Range of line numbers of the refering dataset(will be fed to np.arange()). For nr3d and sr3d, the minimum is 2. For scanrefer, the minimum is 0.")
    parser.add_argument("--line_numbers", type=int, nargs='*', help="When the 'range' parameter is not provided, you can specify non-contiguous line numbers here.")
    parser.add_argument("--ft", type=str, nargs='*', help="List of times in 'yy-mm-dd-HH-MM-SS' format. Requested for result mode.")
    parser.add_argument("--obj-info-ablation-type", type=int, default=0, help="Type of ablation for sr3d. 0: no ablation; 1: no size; 2: min+max; 3: attributes reversed; 4: objects shuffled.")
    parser.add_argument("--use_camera_position", type=bool, default=True)
    parser.add_argument("--filter_behind_obj", type=bool, default=True)

    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    if args.dataset_type == 'nr3d':
        eval_config = confs_nr3d[args.conf_idx]
    elif args.dataset_type == 'sr3d':
        eval_config = confs_sr3d[args.conf_idx]
    elif args.dataset_type == 'scanrefer':
        eval_config = confs_scanrefer[args.conf_idx]
    else:
        print("invalid dataset_type!")

    print("test config:\n", eval_config)
    print("\n")

    system_message = 'Imagine you are an artificial intelligence assistant. You job is to do 3D referring reasoning, namely to find the object for a given utterance from a 3d scene presented as object-centric semantic information.\n'
    system_message += get_system_message() if eval_config['use_code_interpreter'] else ''
    if args.dataset_type == 'sr3d':
        system_message += get_principle_sr3d() if eval_config['use_principle'] else ''
    else:
        system_message += get_principle(eval_config['use_priority']) if eval_config['use_principle'] else ''
    # print('system message:\n',system_message)

    openai_config = {
        'model': eval_config['model'],
        'temperature': 1e-7,
        'top_p': 1e-7,
        # 'max_tokens': 4096,
        'max_tokens': 8192,
        'system_message': system_message,
        # 'load_path': '',
        'save_path': 'chats',
        'debug': True
    }

    tool = eval_config.get('tool')  # scanrefer detection tool
    result_folder_name = eval_config['result_folder_name']

    # set default value for use_object_filter to true in eval_config if not exist
    if not 'use_object_filter' in eval_config:
        eval_config['use_object_filter'] = True

    transcrib3d = Transcrib3D(scannet_data_root=args.scannet_data_root,
                      workspace_path=args.workspace_path,
                      dataset_type=eval_config['dataset_type'],
                      refer_dataset_path=eval_config['refer_dataset_path'],
                      result_folder_name=result_folder_name,
                      gpt_config=openai_config,
                      use_gt_box=eval_config['use_gt_box'],
                      use_principle=eval_config['use_principle'],
                      use_original_viewdep_judge=False,
                      use_object_filter=eval_config['use_object_filter'],
                      scanrefer_tool_name=tool,
                      use_priority=eval_config['use_priority'],
                      use_code_interpreter=eval_config['use_code_interpreter'],

                    #   object_filter_result_check_folder_name="eval_results_scanrefer_4_p_gtbox_valset",
                    #   object_filter_result_check_list=['2023-11-14-08-00-13'],
                      obj_info_ablation_type=args.obj_info_ablation_type
                      )

    ###############################################################################
    if args.mode == 'eval':
        line_number_range = np.arange(args.range[0], args.range[1]) if args.range is not None else args.line_numbers
        transcrib3d.evaluate_on_GPT(line_number_range)  # <---------
    ###############################################################################

    elif args.mode == 'result':
        """analyze results"""
        formatted_time = args.ft
        if isinstance(formatted_time, list):
            print('is list')
            # result_path = ["%s%s/%s.npy" % (result_folder_name, ft, ft) for ft in formatted_time]
            result_path = [os.path.join('results', result_folder_name, ft, f"{ft}.npy") for ft in formatted_time]
        else:
            # result_path = "%s%s/%s.npy" % (result_folder_name, formatted_time, formatted_time) if formatted_time is not None else None
            result_path = os.path.join('results', result_folder_name, formatted_time, f"{formatted_time}.npy") if formatted_time is not None else None
        if result_path:
            # refer3d.analyse_result(result_path)
            config = (os.path.join(transcrib3d.workspace_path, "data"), transcrib3d.use_original_viewdep_judge, transcrib3d.use_gt_box, transcrib3d.scanrefer_iou_thr)
            analyse_result(transcrib3d.dataset_type, transcrib3d.refer_dataset_path, result_path, config)

    elif args.mode == 'self_correct':
        """self correction"""
        # formatted_time="2023-09-11-18-11-35"
        formatted_time = args.ft
        print(formatted_time)
        for time in formatted_time:
            transcrib3d.self_correction_dataset(result_folder_path=args.workspace_path + eval_config['result_folder_name'], formatted_time=time, line_number_list=np.arange(0, 400) if args.dataset_type == 'scanrefer' else np.arange(2, 400))

    elif args.mode == "check_scanrefer":
        """check the how many cases are provided with detected boxes that has 0.5(0.25) or higher iou with gt box"""
        transcrib3d.check_scanrefer_answer_exist_percentage(0.5)


if __name__ == '__main__':
    main()
