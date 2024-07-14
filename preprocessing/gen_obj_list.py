# encoding:utf-8
import numpy as np
import json
from plyfile import PlyData
import pandas as pd
import os
from sklearn.cluster import DBSCAN
from scipy import sparse
from collections import Counter
import argparse
from data.scannet200_constants import VALID_CLASS_IDS_20,CLASS_LABELS_20,VALID_CLASS_IDS_200,CLASS_LABELS_200


def read_dict(file_path):
    with open(file_path) as fin:
        return json.load(fin)
    
def load_scan2cad_meta_data(scan2CAD_meta_file):
    """
    scan2CAD_meta_file: scan2CAD meta-information about the oo-bbox, and 'front-dir'
    """
    scan2CAD_temp = read_dict(scan2CAD_meta_file)

    scan2CAD = dict()
    for scan_id_object_id, object_info in scan2CAD_temp.items():
        # Split the key to get the scan_id and object_id, eg scene0000_00_1
        scan_id = scan_id_object_id[:12]
        object_id = scan_id_object_id[13:]
        # e.g. scan2CAD[("scene0000_00","1")] = object_info
        scan2CAD[(scan_id, object_id)] = object_info
    return scan2CAD

def load_has_front_meta_data(has_front_file):
    """
    Load the has front property (0, 1) of each shapenet object.

    :param has_front_file: The path to the has front annotations file
    :return: dictionary mapping object id -> (0 or 1 'has front property')
    """
    df = pd.read_csv(has_front_file, converters={'syn_id': lambda x: str(x)})
    res = dict()
    for i in range(len(df)):
        yes = df.loc[i]['has_front'] == 1
        if yes:
            res[(str(df.loc[i]['syn_id']), df.loc[i]['model_name'])] = yes
    return res

def gen_obj_list_gt(scan_id, scannet_data_root, referit3d_data_root=None, include_direction_info=False, is_train_data=True):
    # file paths
    aggregation_json_path = os.path.join(scannet_data_root, scan_id, f"{scan_id}_vh_clean.aggregation.json")
    segs_json_path = os.path.join(scannet_data_root, scan_id, f"{scan_id}_vh_clean_2.0.010000.segs.json")
    ply_path = os.path.join(scannet_data_root, scan_id, f"{scan_id}_vh_clean_2.ply")
    ply_align_path = os.path.join(scannet_data_root, scan_id, f"{scan_id}_vh_clean_2.ply")
    axis_align_matrix_path = os.path.join(scannet_data_root, scan_id, f"{scan_id}.txt")
    if include_direction_info:
        object_obb_data_path = os.path.join(referit3d_data_root, 'object_oriented_bboxes_aligned_scans.json')
        object_has_front_data_path = os.path.join(referit3d_data_root, 'shapenet_has_front.csv')

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

    # read in oriented bounding boxes and has_front files
    if include_direction_info:
        obb_data = load_scan2cad_meta_data(object_obb_data_path)
        has_front_data = load_has_front_meta_data(object_has_front_data_path)

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

        # # convert to axis aligned coords
        # vertices_coord = vertices[:,0:3]
        # vertices_transpose = np.vstack([vertices_coord.T,np.ones([1,vertices_coord.shape[0]])])
        # vertices_aligned = np.dot(axis_align_matrix,vertices_transpose)[0:3,:]
        # vertices_aligned = vertices_aligned.T

        # record quantitative information of this object
        x_max, x_min, x_avg = np.max(vertices[:,0]),np.min(vertices[:,0]),np.average(vertices[:,0])
        y_max, y_min, y_avg = np.max(vertices[:,1]),np.min(vertices[:,1]),np.average(vertices[:,1])
        z_max, z_min, z_avg = np.max(vertices[:,2]),np.min(vertices[:,2]),np.average(vertices[:,2])
        # x_max, x_min, x_avg = np.max(vertices_aligned[:,0]), np.min(vertices_aligned[:,0]), np.average(vertices_aligned[:,0])
        # y_max, y_min, y_avg = np.max(vertices_aligned[:,1]), np.min(vertices_aligned[:,1]), np.average(vertices_aligned[:,1])
        # z_max, z_min, z_avg = np.max(vertices_aligned[:,2]), np.min(vertices_aligned[:,2]), np.average(vertices_aligned[:,2])

        info["centroid"] = [x_avg, y_avg, z_avg]
        info["center_position"] = [(x_max+x_min)/2, (y_max+y_min)/2, (z_max+z_min)/2]
        info["size"] = [(x_max-x_min), (y_max-y_min), (z_max-z_min)]
        info["extension"] = [x_min, y_min, z_min, x_max, y_max, z_max]
        info["avg_rgba"] = [int(np.average(vertices[:,3])), int(np.average(vertices[:,4])), int(np.average(vertices[:,5])), int(np.average(vertices[:,6]))]

        if not include_direction_info:
            objects_info.append(info)
            continue

        # read in oriented bounding boxes and has_front information if available
        key = (scan_id, str(obj['id']))
        # 如果在obb_data中能找到，则记录obb信息和旋转角度信息，并尝试记录front信息
        if key in obb_data:
            obb_info = obb_data[key]
            
            if (obb_info['catid_cad'], obb_info['id_cad']) in has_front_data:
                info['has_front'] = True
                info['front_point'] = obb_info['front_point']
            else:
                info['has_front'] = False
                info['front_point'] = obb_info['front_point']

            info['obb'] = obb_info['obj_bbox']
            rot_matrix = np.array(obb_info['obj_rot']) 
            info['rot_angle'] = np.arctan2(rot_matrix[1,0],rot_matrix[0,0]) #x = atan2(sinx,cosx)。

            # # for test
            # front_point = np.array(info['front_point'])
            # center_obb = np.array(info['obb'][0:3])
            # front_vector = front_point-center_obb
            # angle_calc = np.arctan2(front_vector[1],front_vector[0])
            # angle = info['rot_angle']
            # print(key)
            print("%s has front: %s" % (str(key),str(info['has_front'])))
            # print("angle from rot matrix:",angle/np.pi*180)
            # print("angle of front point:",angle_calc/np.pi*180)
            # print("center from referit3d:",center_obb)
            # print("center from ply:",info['center_position'])

        else:
            # print("object not found in %s! %s"%(object_obb_data_path,str(key))) 
            info['has_front'] = False
            info['front_point'] = None
            info['obb'] = None
            info['rot_angle'] = None

        objects_info.append(info)

    return objects_info

def idx_2_label_200(idx):
    print(idx)
    return CLASS_LABELS_200[VALID_CLASS_IDS_200.index(idx)]

def idx_2_label_20(idx):
    print(idx)
    return CLASS_LABELS_20[VALID_CLASS_IDS_20.index(idx)]


def draw_points(points):
    import matplotlib.pyplot as plt

    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, c='b', marker='o')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()


def filter_pointcould(points):
    # use dbscan to filter out outlier points
    dbscan=DBSCAN(eps=0.1,min_samples=20)
    if points.shape[1]==3:
        dbscan.fit(points)
    else:
        dbscan.fit(points[:,0:3])
    counter=Counter(dbscan.labels_)
    main_idx=counter.most_common(2)[0][0]  
    if main_idx==-1:
        main_idx=counter.most_common(2)[-1][0]     
    # print("counter:",counter)
    # print("main_idx:",main_idx)
    points_filtered=points[dbscan.labels_==main_idx]
    return points_filtered

def gen_obj_list_mask3d(scan_id, scannet_data_root, objects_info_folder_name, mask3d_root=None, conf_score_thr=0.5, save_per_instance_points=False, use_200_cats=True, is_train_data=True):
    # file paths
    ply_path = os.path.join(scannet_data_root, scan_id, f"{scan_id}_vh_clean_2.ply")
    ply_align_path = os.path.join(scannet_data_root, scan_id, f"{scan_id}_vh_clean_2.ply")
    # axis_align_matrix_path = os.path.join(scannet_data_root, scan_id, f"{scan_id}.txt")
    scan_info_txt_path = os.path.join(mask3d_root, f"{scan_id}.txt")

    if not os.path.exists(scannet_data_root+objects_info_folder_name):
        os.mkdir(scannet_data_root+objects_info_folder_name)

    # read in Mask3D result .txt file
    with open(scan_info_txt_path)as f:
        object_info_lines=f.readlines()

    # read in the .ply mesh file
    print("loading .ply file...")
    plydata = PlyData.read(ply_path)
    plydata_align = PlyData.read(ply_align_path)

    header = plydata.header
    print("header of .ply file:\n",header)

    # # read in axis aling matrix
    # with open(axis_align_matrix_path, 'r') as file:
    #     lines = file.readlines()
    #     for line in lines:
    #         if "axisAlignment" in line:
    #             numbers = line.split('=')[1].strip().split()
    #             break
    # axis_align_matrix = np.array(numbers, dtype=float).reshape(4, 4)

    # the list to return 
    objects_info = []

    # iterate thru objects
    for line in object_info_lines:
        # get one line of mask3d .txt (one object)
        line_split = line.split()
        print(line_split)

        # mask path, category id and confidential score of the object
        obj_mask_path = line_split[0]
        obj_label_idx = int(line_split[1])
        score = line_split[2]

        # skip objects with low confidential score
        if eval(score) < conf_score_thr:
            continue

        # object id
        obj_id = int(obj_mask_path.split('.')[0].split('_')[-1])
        # category id to categroy label
        if use_200_cats:
            obj_label = idx_2_label_200(obj_label_idx)
        else:
            obj_label = idx_2_label_20(obj_label_idx)

        # read in object mask .txt file to get mask_data (vertices that belong to the object)
        with open(mask3d_root+obj_mask_path) as f:
            mask_lines = f.readlines()
            mask_data =  [int(mask_line.strip()) for mask_line in mask_lines]

        # record info
        info = {"id":obj_id,
              "label":obj_label,
              "score":eval(score)
              }
        
        # find these vertices in ply file
        if is_train_data:
            vertices_all = np.array(plydata_align.elements[0].data) # axis-alined mesh
        else:
            vertices_all = np.array(plydata.elements[0].data) # original mesh
        vertices = vertices_all[np.where(mask_data)]

        vertices = np.array([list(vertex) for vertex in vertices])  # convert to 2-d numpy array

        # # convert to axis aligned coords
        # vertices_coord = vertices[:,0:3]
        # vertices_transpose = np.vstack([vertices_coord.T,np.ones([1,vertices_coord.shape[0]])])
        # vertices_aligned = np.dot(axis_align_matrix,vertices_transpose)[0:3,:]
        # vertices_aligned = vertices_aligned.T

        # filter point cloud and save it
        vertices = filter_pointcould(vertices)

        # save point cloud of each object
        if save_per_instance_points:
            # draw_points(vertices_aligned)
            object_path = os.path.join(scannet_data_root, objects_info_folder_name, f"{scan_id}_{obj_id}_{obj_label}.npy")
            np.save(object_path, vertices, allow_pickle=True)
            print("object points saved to:", object_path)


        # record quantitative information of this object
        x_max,x_min,x_avg = np.max(vertices[:,0]),np.min(vertices[:,0]),np.average(vertices[:,0])
        y_max,y_min,y_avg = np.max(vertices[:,1]),np.min(vertices[:,1]),np.average(vertices[:,1])
        z_max,z_min,z_avg = np.max(vertices[:,2]),np.min(vertices[:,2]),np.average(vertices[:,2])
        # x_max,x_min,x_avg = np.max(vertices_aligned[:,0]),np.min(vertices_aligned[:,0]),np.average(vertices_aligned[:,0])
        # y_max,y_min,y_avg = np.max(vertices_aligned[:,1]),np.min(vertices_aligned[:,1]),np.average(vertices_aligned[:,1])
        # z_max,z_min,z_avg = np.max(vertices_aligned[:,2]),np.min(vertices_aligned[:,2]),np.average(vertices_aligned[:,2])

        info["centroid"] = [x_avg, y_avg, z_avg]
        info["center_position"] = [(x_max+x_min)/2, (y_max+y_min)/2, (z_max+z_min)/2]
        info["size"] = [(x_max-x_min),(y_max-y_min),(z_max-z_min)]
        info["extension"] = [x_min,y_min,z_min,x_max,y_max,z_max]
        info["median_rgba"] = [int(np.median(vertices[:,3])), int(np.median(vertices[:,4])), int(np.median(vertices[:,5])), int(np.median(vertices[:,6]))]

        objects_info.append(info)

    return objects_info

def get_scan_id_list(data_root):
    # 获取scannet数据集中所有场景的名称

    # 获取文件夹下的所有子文件夹名称
    subfolders = [subfolder for subfolder in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, subfolder))]

    # 打印子文件夹名称
    print("Subfolders:", subfolders)

    print('objects_info' in subfolders)

    scan_id_list=[]
    for name in subfolders:
        if 'scene' in name:
            scan_id_list.append(name)
    print("%d scan ids found." %len(scan_id_list))
    return scan_id_list

if __name__=='__main__':
    # parse ScanNet download path
    parser = argparse.ArgumentParser(description="Generate object information and save to .npy file.")
    parser.add_argument("--scannet_download_path", type=str, help="Path of the ScanNet data download folder. There should be subfolder 'scans' under it.")
    parser.add_argument("--bbox_type", type=str, choices=['gt', 'mask3d'], default='gt', help="Type of generated bounding boxes. 'gt' means using groud truth segmentation data provided by ScanNet, while 'mask3d' means using Mask3D to segment objects.")
    parser.add_argument("--include_direction", action="store_true", help="Whether to include direction data in the generated object information.")
    parser.add_argument("--referit3d_data_path", type=str, default="None", help="Path of the ReferIt3D data download folder.")
    parser.add_argument("--mask3d_result_path", type=str, default="None", help="Path of the mask result folder. It should look like xxx/Mask3D/eval_output/instance_evaluation_mask3d_export_scannet200_0/val/")
    parser.add_argument("--mask3d_20_cats", action="store_true", help="If set, will use 20 categories for Mask3D. 200 if not set.")
    args = parser.parse_args()

    scannet_download_path = args.scannet_download_path
    bbox_type = args.bbox_type
    include_direction = args.include_direction
    referit3d_data_path = args.referit3d_data_path
    mask3d_result_path = args.mask3d_result_path
    mask3d_20_cats = args.mask3d_20_cats

    if bbox_type == "gt":
        # process scannet train set
        scannet_data_root = os.path.join(scannet_download_path, 'scans')
        scan_id_list_train = get_scan_id_list(scannet_data_root)
        save_dir = save_path = os.path.join(scannet_data_root, 'objects_info')

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        for idx,scan_id in enumerate(scan_id_list_train):
            print("Processing train set: %s (%d/%d)..." % (scan_id,idx+1,len(scan_id_list_train)))
            objects_info = gen_obj_list_gt(scan_id=scan_id, 
                                        scannet_data_root=scannet_data_root,  
                                        referit3d_data_root=referit3d_data_path,
                                        include_direction_info=include_direction,
                                        is_train_data=True
                                        )
            save_path = os.path.join(save_dir, f"objects_info_{scan_id}.npy")
            np.save(save_path, objects_info, allow_pickle=True)
            print("Object information saved to:", save_path)

        # process scannet test set
        scannet_data_root = os.path.join(scannet_download_path, 'scans_test')
        scan_id_list_test = get_scan_id_list(scannet_data_root)
        save_dir = os.path.join(scannet_data_root, 'objects_info')

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        for idx,scan_id in enumerate(scan_id_list_test):
            print("Processing test set: %s (%d/%d)..." % (scan_id,idx+1,len(scan_id_list_test)))
            objects_info = gen_obj_list_gt(scan_id=scan_id, 
                                        scannet_data_root=scannet_data_root,  
                                        referit3d_data_root=referit3d_data_path,
                                        include_direction_info=include_direction,
                                        is_train_data=False
                                        )
            save_path = os.path.join(save_dir, f"objects_info_{scan_id}.npy")
            np.save(save_path, objects_info, allow_pickle=True)
            print("Object information saved to:", save_path)

    elif bbox_type == 'mask3d':
        ## mask3d ##
        use_200_cats = not mask3d_20_cats

        # process scannet val set, which is defined in data/scannetv2_val.txt
        scannet_data_root = os.path.join(scannet_download_path, 'scans')
        scan_id_list_train=get_scan_id_list(scannet_data_root)

        with open('data/scannetv2_val.txt') as f:
            scan_id_list_val=f.readlines()
        for idx,line in enumerate(scan_id_list_val):
            scan_id_list_val[idx]=line.strip()

        objects_info_folder_name="objects_info_mask3d_200c" if use_200_cats else "objects_info_mask3d_20c"
        save_dir = os.path.join(scannet_data_root, objects_info_folder_name)

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        for idx,scan_id in enumerate(scan_id_list_val):
            print("Processing train set: %s (%d/%d)..." % (scan_id,idx+1,len(scan_id_list_val)))

            conf_score_thr=0.2
            objects_info=gen_obj_list_mask3d(scan_id=scan_id, 
                               scannet_data_root=scannet_data_root, 
                               objects_info_folder_name=objects_info_folder_name,
                               mask3d_root=mask3d_result_path,
                               conf_score_thr=conf_score_thr,
                               save_per_instance_points=False,
                               use_200_cats=use_200_cats
                               )

            save_path = os.path.join(save_dir, f"{objects_info_folder_name}_{scan_id}.npy")
            np.save(save_path, objects_info, allow_pickle=True)
            print("Object information saved to:", save_path)
