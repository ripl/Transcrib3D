import numpy as np
import json
from plyfile import PlyData
import pandas as pd
import os

def gen_data_pair(scan_id, scannet_data_root, is_train_data=False, use_high_res_seg=False):
    ### about low/high res segmentation: https://github.com/ScanNet/ScanNet
    # file paths
    if use_high_res_seg:
        ply_path = os.path.join(scannet_data_root, scan_id, f"{scan_id}_vh_clean.ply")
        aggregation_json_path = os.path.join(scannet_data_root, scan_id, f"{scan_id}_vh_clean.aggregation.json")
        segs_json_path = os.path.join(scannet_data_root, scan_id, f"{scan_id}_vh_clean.segs.json")
    else:
        ply_path = os.path.join(scannet_data_root, scan_id, f"{scan_id}_vh_clean_2.ply")
        aggregation_json_path = os.path.join(scannet_data_root, scan_id, f"{scan_id}.aggregation.json")
        segs_json_path = os.path.join(scannet_data_root, scan_id, f"{scan_id}_vh_clean_2.0.010000.segs.json")
    print(aggregation_json_path)
    print(segs_json_path)
    # aggregation_json_path = os.path.join(scannet_data_root, scan_id, f"{scan_id}.aggregation.json")
    # segs_json_path = os.path.join(scannet_data_root, scan_id, f"{scan_id}_vh_clean_2.0.010000.segs.json")
    # ply_path = os.path.join(scannet_data_root, scan_id, f"{scan_id}_vh_clean_2.ply")
    ply_align_path = os.path.join(scannet_data_root, scan_id, f"{scan_id}_vh_clean_2_aligned.ply")
    axis_align_matrix_path = os.path.join(scannet_data_root, scan_id, f"{scan_id}.txt")

    # open the .json file that records vertex semantics (which segmentation set does a vertex belong to)
    print("loading .segs.json file...")
    with open(segs_json_path,'r') as vertex_seg_jfile:
        seg_result = json.load(vertex_seg_jfile)["segIndices"]
        seg_result = np.array(seg_result)
    
    # open the .json file that records object information (which segmentation sets does this object possess.)
    print("loading .aggregation.json file...")
    with open(aggregation_json_path,'r') as vertex_agg_jfile:
        objects = json.load(vertex_agg_jfile)["segGroups"] # a list of dicts

    # read in the .ply mesh file
    print("loading .ply file...")
    plydata = PlyData.read(ply_path)
    plydata_align = PlyData.read(ply_align_path)

    header = plydata.header
    print("header of .ply file:\n",header)

    # read in axis aling matrix
    with open(axis_align_matrix_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            if "axisAlignment" in line:
                numbers = line.split('=')[1].strip().split()
                break
    axis_align_matrix = np.array(numbers, dtype=float).reshape(4, 4)

    # the list to return 
    objects_info = []

    # iterate thru objects
    for obj in objects:
        # object id and label
        info = {"id":obj["id"],
              "label":obj["label"]}
        
        # segmentation sets that belong to the object
        seg_index = obj["segments"]

        # vertices that belong to the object
        vertices_index = np.where(np.in1d(seg_result, seg_index))

        # find these vertices in ply file
        if is_train_data:
            vertices_all = np.array(plydata_align.elements[0].data) # axis-alined mesh
        else:
            vertices_all = np.array(plydata.elements[0].data) # original mesh
        vertices = vertices_all[vertices_index]
        vertices = np.array([list(vertex) for vertex in vertices]) # convert to 2-d numpy array

        info["vertices"] = vertices

        objects_info.append(info)

    return objects_info
    
def get_scan_id_list(data_root):
    # get scan_ids of ScanNet by searching subfolder names
    subfolders = [subfolder for subfolder in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, subfolder))]
    print("Subfolders:", subfolders)

    scan_id_list=[]
    for name in subfolders:
        if 'scene' in name:
            scan_id_list.append(name)
    print("%d scan ids found." %len(scan_id_list))
    return scan_id_list


if __name__ == "__main__":
    # scannet_data_root = os.path.join('H:\ScanNet_Data\data\scannet\scans', 'scans')
    scannet_data_root = "H:\ScanNet_Data\data\scannet\scans"
    scan_id_list_train = get_scan_id_list(scannet_data_root)
    use_high_res_seg = True
    save_dir = save_path = os.path.join('data', 'pointcloud_label_data_hi-res') if use_high_res_seg else os.path.join('data', 'pointcloud_label_data') 

    print(scan_id_list_train)
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    print_label = True

    # interate thru scan_id list
    for scan_id in scan_id_list_train:
        # make new result directory for a scan
        scan_save_dir = os.path.join(save_dir, scan_id)
        if not os.path.exists(scan_save_dir):
            os.makedirs(scan_save_dir)

        # generate data pair
        objects_info = gen_data_pair(scan_id, scannet_data_root, use_high_res_seg=use_high_res_seg)

        # save data
        for obj_info in objects_info:
            obj_id, obj_label, vertices = obj_info['id'], obj_info['label'], obj_info['vertices']

            if print_label is True:
                print("obj_id:", obj_id)
                print("obj_label:", obj_label)
                print("vertices:", vertices)
                print_label = False

            # save pointcloud 
            pointcloud_path = os.path.join(scan_save_dir, f"{scan_id}_{obj_id}_pointcloud.npy")
            np.save(pointcloud_path, vertices)
            # save label
            label_path = os.path.join(scan_save_dir, f"{scan_id}_{obj_id}_label.txt")
            with open(label_path, 'w', encoding='utf-8') as file:
                file.write(obj_label)
