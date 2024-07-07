import numpy as np
import csv
import json
import os
from utils.utils import *

def load_refer_dataset(transcrib3d, line_numbers=[2,]):
    # load the refering dataset from the corresponding file,
    # the dataset is one of (sr3d, nr3d, scanrefer).
    # and check if the line numbers is in available range.
    if transcrib3d.dataset_type == 'sr3d':
        transcrib3d.sr3d_data = read_csv_with_index(transcrib3d.refer_dataset_path)
        assert np.max(line_numbers) <= len(transcrib3d.sr3d_data) + 1, "line number %d > %d!" % (np.max(line_numbers), len(transcrib3d.sr3d_data) + 1)
        assert np.min(line_numbers) >= 2, "sr3d line number %s < 2!" % np.min(line_numbers)
        return transcrib3d.sr3d_data
    elif transcrib3d.dataset_type == 'nr3d':
        transcrib3d.nr3d_data = read_csv_with_index(transcrib3d.refer_dataset_path)
        assert np.max(line_numbers) <= len(transcrib3d.nr3d_data) + 1, "line number %d > %d!" % (np.max(line_numbers), len(transcrib3d.nr3d_data) + 1)
        assert np.min(line_numbers) >= 2, "nr3d line number %s < 2!" % np.min(line_numbers)
        return transcrib3d.nr3d_data
    elif transcrib3d.dataset_type == 'scanrefer':
        transcrib3d.scanrefer_data = read_json(transcrib3d.refer_dataset_path)
        assert np.max(line_numbers) <= len(transcrib3d.scanrefer_data) - 1, "line number %d > %d!" % (np.max(line_numbers), len(transcrib3d.scanrefer_data) - 1)
        assert np.min(line_numbers) >= 0, "scanrefer description number %s < 0!" % np.min(line_numbers)
        return transcrib3d.scanrefer_data
    else:
        print("Invalid dataset!")
        return None
    
def load_refer_dataset_pure(dataset_type:str, refer_dataset_path:str):
    # load the refering dataset from the corresponding file,
    # the dataset is one of (sr3d, nr3d, scanrefer).
    if dataset_type in ['sr3d', 'nr3d', 'scanrefer']:
        return read_csv_with_index(refer_dataset_path)
    else:
        print("Invalid dataset!")
        return None


def read_csv_with_index(file_path):
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

def read_json(file_path):
    # read in the data of scanrefer(.json)，returning a list of dictionary(same as that in the json file). index starts from 0.
    with open(file_path, 'r') as jf:
        jf_data = jf.read()  # jf_data is a string
        data = json.loads(jf_data)
    return data

def get_scanrefer_camera_info_aligned(data_root, scan_id, object_id, annotation_id):
    """Return camera info which is axis aligned"""
    json_path = os.path.join(data_root, 'scanrefer_camera_info', '%s.anns.json' % scan_id)
    with open(json_path) as f:
        data = json.load(f)
    target_annotation = [d for d in data if d['object_id'] == object_id and d['ann_id'] == annotation_id]
    if target_annotation:
        if len(target_annotation) > 1:
            print("%d camera info found! Returning the first one. scan_id=%s, object_id=%s, annotation_id=%s." % (len(target_annotation), scan_id, object_id, annotation_id))
        camera_info = target_annotation[0]['camera']
        position = camera_info['position']
        lookat = camera_info['lookat']
        # align the vectors(coordinates)
        axis_align_matrix = get_axis_align_matrix(os.path.join(data_root, "scannet_scene_info", scan_id + ".txt"))
        position_aligned = np.dot(axis_align_matrix, np.append(position, 1).reshape(4, 1))[0:3].reshape(-1)
        lookat_aligned = np.dot(axis_align_matrix, np.append(lookat, 1).reshape(4, 1))[0:3].reshape(-1)
        camera_info_aligned = {'position': position_aligned, 'lookat': lookat_aligned}
        return camera_info_aligned
    else:
        print("No camera info found! scan_id=%s, object_id=%s, annotation_id=%s." % (scan_id, object_id, annotation_id))
        return None
    
def get_axis_align_matrix(txt_path):
    with open(txt_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            if "axisAlignment" in line:
                numbers = line.split('=')[1].strip().split()
                break
    axis_align_matrix = np.array(numbers, dtype=float).reshape(4, 4)
    return axis_align_matrix

def get_scanrefer_gt_box(scan_id, object_id):
    """
    Convert a bounding box in center-size format to extension format.

    Parameters:
        box_center_size (list): A box in in center-size format: [cx, cy, cz, sx, sy, sz].

    Returns:
        list: A box in in extension format: [xmin, ymin, zmin, xmax, ymax, zmax]

    Examples:
        A cube box at [0, 0, 0] with side length 1 -- [0, 0, 0, 1, 1, 1] will be converted to [-1, -1, -1, 1, 1, 1]
    """
    # get the ground truth bounding box according to scan_id and object id
    # from file scan_id_aligned_bbox.npy, which could be produced in pre-process of ScanRefer repo.
    # scan_id_aligned_bbox.npy has matrices of shape (N, 8)，with each row as a box. box format is (cx,cy,cz,sx,sy,sz,label_id,obj_id).
    gt_box_path = "/share/data/ripl/vincenttann/ScanRefer/data/scannet/scannet_data/" + scan_id + "_aligned_bbox.npy"
    gt_boxes = np.load(gt_box_path)
    gt_box = gt_boxes[gt_boxes[:, -1].reshape(-1).astype(int) == int(object_id)]
    assert len(gt_box) > 0, "No gt box found!!! scan_id=%d, object_id=%d" % (scan_id, object_id)
    assert len(gt_box) == 1, "Multiple gt box found!!! scan_id=%d, object_id=%d, gt_box found:%s" % (scan_id, object_id, str(gt_box))
    return center_size_to_extension(gt_box.reshape(-1)[0:6])