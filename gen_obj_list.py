# encoding:utf-8
import numpy as np
import json
from plyfile import PlyData
import pandas as pd
import os
from sklearn.cluster import DBSCAN
from scipy import sparse
from collections import Counter
from scannet200_constants import VALID_CLASS_IDS_20,CLASS_LABELS_20,VALID_CLASS_IDS_200,CLASS_LABELS_200


def read_dict(file_path):
    # 将json文件读入为字典
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

def draw_points(points):
    import matplotlib.pyplot as plt
    # 提取x、y和z坐标
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    # 创建一个3D散点图
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 绘制散点图
    ax.scatter(x, y, z, c='b', marker='o')

    # 设置坐标轴标签
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # 显示图形
    plt.show()

def save_points(points):
    np.save("/share/data/ripl/scannet_raw")

def gen_obj_list(scan_id, data_root, referit3d_data_root):
    # 各个文件的路径
    aggregation_json_path = data_root+scan_id+"/"+scan_id+"_vh_clean.aggregation.json"
    segs_json_path = data_root+scan_id+"/"+scan_id+"_vh_clean_2.0.010000.segs.json"
    ply_path = data_root+scan_id+"/"+scan_id+"_vh_clean_2.ply"
    object_obb_data_path=referit3d_data_root+'object_oriented_bboxes_aligned_scans.json'
    object_has_front_data_path=referit3d_data_root+'shapenet_has_front.csv'

    # 打开记录vertex语义（所述分割集）的json文件
    print("loading .segs.json file...")
    with open(segs_json_path,'r') as vertex_seg_jfile:
        seg_result=json.load(vertex_seg_jfile)["segIndices"]
        seg_result=np.array(seg_result)
    
    # 打开记录object信息的json文件
    print("loading .aggregation.json file...")
    with open(aggregation_json_path,'r') as vertex_agg_jfile:
        objects=json.load(vertex_agg_jfile)["segGroups"] #dict的list

    # 读入ply文件
    print("loading .ply file...")
    plydata=PlyData.read(ply_path)

    header=plydata.header
    print("header of .ply file:",header)

    # 读入axis aline矩阵
    filename = data_root+scan_id+'/'+scan_id+'.txt'  # 替换成实际的文件名
    with open(filename, 'r') as file:
        lines=file.readlines()
        for line in lines:
            if "axisAlignment" in line:
                numbers = line.split('=')[1].strip().split()
                break
    axis_align_matrix = np.array(numbers, dtype=float).reshape(4, 4)

    # 读入obb和has_front文件
    obb_data=load_scan2cad_meta_data(object_obb_data_path)
    has_front_data=load_has_front_meta_data(object_has_front_data_path)

    # 定义最终返回的，记录所有object信息的dict
    objects_info=[]

    # 遍历objects
    for obj in objects:
        # 记录物体基本信息
        info={"id":obj["id"],
              "label":obj["label"]}
        
        # 获取物体所包含的所有分割集编号
        seg_index=obj["segments"]

        # 获取物体所包含的所有vertex的编号
        vertices_index=np.where(np.in1d(seg_result,seg_index)) #np.in1d函数在seg_result中查找每个vertex的分割集编号是否在seg_indx中
        # print(vertices_index)

        # 在ply文件中找到物体对应的vertices
        vertices_all=np.array(plydata.elements[0].data)
        vertices=vertices_all[vertices_index]
        vertices=np.array([list(vertex) for vertex in vertices]) #转换为2维numpy array
        # print(vertices)
        # print(vertices.shape)

        # 转为axis aligned坐标
        vertices_coord=vertices[:,0:3]
        vertices_transpose=np.vstack([vertices_coord.T,np.ones([1,vertices_coord.shape[0]])])
        vertices_aligned=np.dot(axis_align_matrix,vertices_transpose)[0:3,:]
        vertices_aligned=vertices_aligned.T

        # 计算相关参数并记录定量信息
        # x_max,x_min,x_avg=np.max(vertices[:,0]),np.min(vertices[:,0]),np.average(vertices[:,0])
        # y_max,y_min,y_avg=np.max(vertices[:,1]),np.min(vertices[:,1]),np.average(vertices[:,1])
        # z_max,z_min,z_avg=np.max(vertices[:,2]),np.min(vertices[:,2]),np.average(vertices[:,2])
        x_max,x_min,x_avg=np.max(vertices_aligned[:,0]),np.min(vertices_aligned[:,0]),np.average(vertices_aligned[:,0])
        y_max,y_min,y_avg=np.max(vertices_aligned[:,1]),np.min(vertices_aligned[:,1]),np.average(vertices_aligned[:,1])
        z_max,z_min,z_avg=np.max(vertices_aligned[:,2]),np.min(vertices_aligned[:,2]),np.average(vertices_aligned[:,2])

        # info["quan_info"]=[x_avg,y_avg,z_avg,(x_max-x_min),(y_max-y_min),(z_max-z_min)]
        info["centroid"]=[x_avg, y_avg, z_avg]
        info["center_position"]=[(x_max+x_min)/2, (y_max+y_min)/2, (z_max+z_min)/2]
        info["size"]=[(x_max-x_min),(y_max-y_min),(z_max-z_min)]
        info["extension"]=[x_min,y_min,z_min,x_max,y_max,z_max]
        info["avg_rgba"]=[int(np.average(vertices[:,3])), int(np.average(vertices[:,4])), int(np.average(vertices[:,5])), int(np.average(vertices[:,6]))]

        # 读取oriented bbox 和 方向信息
        key=(scan_id,str(obj['id']))
        # 如果在obb_data中能找到，则记录obb信息和旋转角度信息，并尝试记录front信息
        if key in obb_data:
            obb_info=obb_data[key]
            

            if (obb_info['catid_cad'], obb_info['id_cad']) in has_front_data:
                info['has_front']=True
                info['front_point']=obb_info['front_point']
            else:
                info['has_front']=False
                info['front_point']=obb_info['front_point']

            info['obb']=obb_info['obj_bbox']
            rot_matrix=np.array(obb_info['obj_rot']) 
            info['rot_angle']=np.arctan2(rot_matrix[1,0],rot_matrix[0,0]) #用x=atan2(sinx,cosx)。
            # rot 4x4太大了没必要全记录，测试了几个都是front_point算出来的角度+180

            # # for test
            # front_point=np.array(info['front_point'])
            # center_obb=np.array(info['obb'][0:3])
            # front_vector=front_point-center_obb
            # angle_calc=np.arctan2(front_vector[1],front_vector[0])
            # angle=info['rot_angle']
            # print(key)
            print("%s has front: %s" % (str(key),str(info['has_front'])))
            # print("angle from rot matrix:",angle/np.pi*180)
            # print("angle of front point:",angle_calc/np.pi*180)
            # print("center from referit3d:",center_obb)
            # print("center from ply:",info['center_position'])


        else:
            print("object not found in %s! %s"%(object_obb_data_path,str(key))) 
            info['has_front']=False
            info['front_point']=None
            info['obb']=None
            info['rot_angle']=None

        objects_info.append(info)

    return objects_info

def gen_obj_list_group_free(scan_id,scannet_root,group_free_root):
    # 定义路径
    ply_path = scannet_root+scan_id+"/"+scan_id+"_vh_clean_2_aligned.ply"
    group_free_box_path=group_free_root+scan_id+".npy"
    
    # 读入ply文件
    print("loading .ply file...")
    plydata=PlyData.read(ply_path)
    print("header:",plydata.header)

    # 在ply文件中找到物体对应的vertices
    vertices=np.array(plydata.elements[0].data)
    vertices=np.array([list(vertex) for vertex in vertices]) #转换为2维numpy array
    vertices_coord=vertices[:,0:3]
    vertices_rgba=vertices[:,3:]
    
    # 读入gf bounding box信息
    gf_data=np.load(group_free_box_path,allow_pickle=True).reshape(-1)[0]
    gf_classes=gf_data['class']
    gf_boxes=gf_data['box']
    assert len(gf_classes)==len(gf_boxes), "len(gf_classes) != len(gf_boxes): %d != %d"%(len(gf_classes),len(gf_boxes))

    # 定义最终返回的，记录所有object信息的dict的list
    objects_info=[]

    # 遍历objects
    for idx,box in enumerate(gf_boxes):
        # 记录物体基本信息
        info={"id":idx,
              "label":gf_classes[idx]}
        
        # 读入group free的bounding box
        xmin,ymin,zmin,xmax,ymax,zmax=box

        # 在ply文件中找到处于这个bounding box内的点
        # 使用向量化操作筛选点
        inside_x = np.logical_and(vertices_coord[:, 0] >= xmin, vertices_coord[:, 0] <= xmax)
        inside_y = np.logical_and(vertices_coord[:, 1] >= ymin, vertices_coord[:, 1] <= ymax)
        inside_z = np.logical_and(vertices_coord[:, 2] >= zmin, vertices_coord[:, 2] <= zmax)
        # 组合三个条件
        inside_mask = np.logical_and.reduce((inside_x, inside_y, inside_z))
        inside_count=np.sum(inside_mask)
        if inside_count<2:
            print("Invalid bounding box: idx=%d, inside_count=%d"%(idx,inside_count))
            continue
        # 筛选出在边界框内的坐标点
        coords_inside = vertices_coord[inside_mask]
        # print(coords_inside.shape)
        rgba_inside = vertices_rgba[inside_mask]

        # 计算相关参数并记录定量信息
        x_max,x_min,x_avg=np.max(coords_inside[:,0]),np.min(coords_inside[:,0]),np.average(coords_inside[:,0])
        y_max,y_min,y_avg=np.max(coords_inside[:,1]),np.min(coords_inside[:,1]),np.average(coords_inside[:,1])
        z_max,z_min,z_avg=np.max(coords_inside[:,2]),np.min(coords_inside[:,2]),np.average(coords_inside[:,2])

        info["centroid"]=[x_avg, y_avg, z_avg]
        info["center_position"]=[(x_max+x_min)/2, (y_max+y_min)/2, (z_max+z_min)/2]
        info["size"]=[(x_max-x_min),(y_max-y_min),(z_max-z_min)]
        info["extension"]=[x_min,y_min,z_min,x_max,y_max,z_max]
        info["avg_rgba"]=[int(np.average(rgba_inside[:,0])), int(np.average(rgba_inside[:,1])), int(np.average(rgba_inside[:,2])), int(np.average(rgba_inside[:,3]))]
        info["median_rgba"]=[int(np.median(rgba_inside[:,0])), int(np.median(rgba_inside[:,1])), int(np.median(rgba_inside[:,2])), int(np.median(rgba_inside[:,3]))]

        # 记录方向信息，暂时没有
        info['has_front']=False
        info['front_point']=None
        info['obb']=None
        info['rot_angle']=None
        
        # 最后记录内部所有点的点云，可能用于后续分析
        # info["pc"]=coords_inside
        info["pc"]=np.hstack([coords_inside,rgba_inside])


        objects_info.append(info)

    return objects_info


def idx_2_label_200(idx):
    print(idx)
    return CLASS_LABELS_200[VALID_CLASS_IDS_200.index(idx)]

def idx_2_label_20(idx):
    print(idx)
    return CLASS_LABELS_20[VALID_CLASS_IDS_20.index(idx)]

def filter_pointcould(points):
    # 使用dbscan找到主要部分
    dbscan=DBSCAN(eps=0.1,min_samples=20)
    if points.shape[1]==3:
        dbscan.fit(points)
    else:
        dbscan.fit(points[:,0:3])
    counter=Counter(dbscan.labels_)
    main_idx=counter.most_common(2)[0][0]  
    # 如果是-1最多，那取其余中最多的cluster
    if main_idx==-1:
        main_idx=counter.most_common(2)[-1][0]     
    print("counter:",counter)
    print("main_idx:",main_idx)
    points_filtered=points[dbscan.labels_==main_idx]
    return points_filtered

def gen_obj_list_mask3d(scan_id,scannet_root='/share/data/ripl/scannet_raw/test/',mask3d_root=None,referit3d_data_root=None):
    # 各个文件的路径
    # aggregation_json_path = scannet_root+scan_id+"/"+scan_id+"_vh_clean.aggregation.json"
    # segs_json_path = scannet_root+scan_id+"/"+scan_id+"_vh_clean_2.0.010000.segs.json"
    ply_path = scannet_root+scan_id+"/"+scan_id+"_vh_clean_2.ply"
    object_obb_data_path=referit3d_data_root+'object_oriented_bboxes_aligned_scans.json'
    object_has_front_data_path=referit3d_data_root+'shapenet_has_front.csv'

    scan_info_txt_path=mask3d_root+scan_id+'.txt'

    # 打开txt
    with open(scan_info_txt_path)as f:
        object_info_lines=f.readlines()


    # 读入ply文件
    print("loading .ply file...")
    plydata=PlyData.read(ply_path)

    header=plydata.header
    print("header of .ply file:",header)

    # # 读入axis aline矩阵
    # filename = scannet_root+scan_id+'/'+scan_id+'.txt'  # 替换成实际的文件名
    # with open(filename, 'r') as file:
    #     lines=file.readlines()
    #     for line in lines:
    #         if "axisAlignment" in line:
    #             numbers = line.split('=')[1].strip().split()
    #             break
    # axis_align_matrix = np.array(numbers, dtype=float).reshape(4, 4)

    # 读入obb和has_front文件
    obb_data=load_scan2cad_meta_data(object_obb_data_path)
    has_front_data=load_has_front_meta_data(object_has_front_data_path)

    # 定义最终返回的，记录所有object信息的dict
    objects_info=[]

    # 遍历objects
    for line in object_info_lines:
        line_split=line.split()
        print(line_split)
        obj_mask_path=line_split[0]
        obj_id=int(obj_mask_path.split('.')[0].split('_')[-1])

        obj_label_idx=int(line_split[1])
        obj_label=idx_2_label_200(obj_label_idx)

        with open(mask3d_root+obj_mask_path) as f:
            mask_lines=f.readlines()
            mask_data= [int(mask_line.strip()) for mask_line in mask_lines]

        # 记录物体基本信息
        info={"id":obj_id,
              "label":obj_label}
        
        # 在ply文件中找到物体对应的vertices
        vertices_all=np.array(plydata.elements[0].data)
        vertices=vertices_all[np.where(mask_data)]

        vertices=np.array([list(vertex) for vertex in vertices]) #转换为2维numpy array
        # print(vertices)
        # print(vertices.shape)

        # 转为axis aligned坐标
        vertices_coord=vertices[:,0:3]
        # vertices_transpose=np.vstack([vertices_coord.T,np.ones([1,vertices_coord.shape[0]])])
        # vertices_aligned=np.dot(axis_align_matrix,vertices_transpose)[0:3,:]
        # vertices_aligned=vertices_aligned.T

        # 计算相关参数并记录定量信息
        # x_max,x_min,x_avg=np.max(vertices[:,0]),np.min(vertices[:,0]),np.average(vertices[:,0])
        # y_max,y_min,y_avg=np.max(vertices[:,1]),np.min(vertices[:,1]),np.average(vertices[:,1])
        # z_max,z_min,z_avg=np.max(vertices[:,2]),np.min(vertices[:,2]),np.average(vertices[:,2])
        x_max,x_min,x_avg=np.max(vertices_coord[:,0]),np.min(vertices_coord[:,0]),np.average(vertices_coord[:,0])
        y_max,y_min,y_avg=np.max(vertices_coord[:,1]),np.min(vertices_coord[:,1]),np.average(vertices_coord[:,1])
        z_max,z_min,z_avg=np.max(vertices_coord[:,2]),np.min(vertices_coord[:,2]),np.average(vertices_coord[:,2])

        # info["quan_info"]=[x_avg,y_avg,z_avg,(x_max-x_min),(y_max-y_min),(z_max-z_min)]
        info["centroid"]=[x_avg, y_avg, z_avg]
        info["center_position"]=[(x_max+x_min)/2, (y_max+y_min)/2, (z_max+z_min)/2]
        info["size"]=[(x_max-x_min),(y_max-y_min),(z_max-z_min)]
        info["extension"]=[x_min,y_min,z_min,x_max,y_max,z_max]
        info["avg_rgba"]=[int(np.average(vertices[:,3])), int(np.average(vertices[:,4])), int(np.average(vertices[:,5])), int(np.average(vertices[:,6]))]

        # 读取oriented bbox 和 方向信息
        key=(scan_id,str(obj_id))
        # 如果在obb_data中能找到，则记录obb信息和旋转角度信息，并尝试记录front信息
        if key in obb_data:
            obb_info=obb_data[key]
            

            if (obb_info['catid_cad'], obb_info['id_cad']) in has_front_data:
                info['has_front']=True
                info['front_point']=obb_info['front_point']
            else:
                info['has_front']=False
                info['front_point']=obb_info['front_point']

            info['obb']=obb_info['obj_bbox']
            rot_matrix=np.array(obb_info['obj_rot']) 
            info['rot_angle']=np.arctan2(rot_matrix[1,0],rot_matrix[0,0]) #用x=atan2(sinx,cosx)。
            # rot 4x4太大了没必要全记录，测试了几个都是front_point算出来的角度+180

            # # for test
            # front_point=np.array(info['front_point'])
            # center_obb=np.array(info['obb'][0:3])
            # front_vector=front_point-center_obb
            # angle_calc=np.arctan2(front_vector[1],front_vector[0])
            # angle=info['rot_angle']
            # print(key)
            print("%s has front: %s" % (str(key),str(info['has_front'])))
            # print("angle from rot matrix:",angle/np.pi*180)
            # print("angle of front point:",angle_calc/np.pi*180)
            # print("center from referit3d:",center_obb)
            # print("center from ply:",info['center_position'])


        else:
            print("object not found in %s! %s"%(object_obb_data_path,str(key))) 
            info['has_front']=False
            info['front_point']=None
            info['obb']=None
            info['rot_angle']=None

        objects_info.append(info)

    return objects_info


def gen_obj_list_mask3d_align(scan_id,scannet_root,objects_info_folder_name,mask3d_root=None,referit3d_data_root=None,conf_score_thr=0.5,save_per_instance_points=False,use_200_cats=True):
    # 各个文件的路径
    # aggregation_json_path = scannet_root+scan_id+"/"+scan_id+"_vh_clean.aggregation.json"
    # segs_json_path = scannet_root+scan_id+"/"+scan_id+"_vh_clean_2.0.010000.segs.json"
    ply_path = scannet_root+scan_id+"/"+scan_id+"_vh_clean_2.ply"
    object_obb_data_path=referit3d_data_root+'object_oriented_bboxes_aligned_scans.json'
    object_has_front_data_path=referit3d_data_root+'shapenet_has_front.csv'

    scan_info_txt_path=mask3d_root+scan_id+'.txt'

    if not os.path.exists(scannet_root+objects_info_folder_name):
        os.mkdir(scannet_root+objects_info_folder_name)

    # 打开txt
    with open(scan_info_txt_path)as f:
        object_info_lines=f.readlines()


    # 读入ply文件
    print("loading .ply file...")
    plydata=PlyData.read(ply_path)

    header=plydata.header
    print("header of .ply file:",header)

    # 读入axis aline矩阵
    filename = scannet_root+scan_id+'/'+scan_id+'.txt'  # 替换成实际的文件名
    with open(filename, 'r') as file:
        lines=file.readlines()
        for line in lines:
            if "axisAlignment" in line:
                numbers = line.split('=')[1].strip().split()
                break
    axis_align_matrix = np.array(numbers, dtype=float).reshape(4, 4)

    # 读入obb和has_front文件
    obb_data=load_scan2cad_meta_data(object_obb_data_path)
    has_front_data=load_has_front_meta_data(object_has_front_data_path)

    # 定义最终返回的，记录所有object信息的dict
    objects_info=[]

    # 遍历objects
    for line in object_info_lines:
        # 读入mask3d结果txt文件的一行（一个物体）
        line_split=line.split()
        print(line_split)
        # 该物体的mask的路径、category id和confidential score
        obj_mask_path=line_split[0]
        obj_label_idx=int(line_split[1])
        score=line_split[2]

        # 跳过得分小于thr的
        if eval(score)<conf_score_thr:
            continue

        # 从路径中提取id
        obj_id=int(obj_mask_path.split('.')[0].split('_')[-1])
        # 物体的category id to categroy label
        if use_200_cats:
            obj_label=idx_2_label_200(obj_label_idx)
        else:
            obj_label=idx_2_label_20(obj_label_idx)

        # 打开object mask txt文件，定义mask_data
        with open(mask3d_root+obj_mask_path) as f:
            mask_lines=f.readlines()
            mask_data= [int(mask_line.strip()) for mask_line in mask_lines]

        # 记录物体基本信息
        info={"id":obj_id,
              "label":obj_label,
              "score":eval(score)
              }
        
        # 在ply文件中找到物体对应的vertices
        vertices_all=np.array(plydata.elements[0].data)
        vertices=vertices_all[np.where(mask_data)]

        vertices=np.array([list(vertex) for vertex in vertices]) #转换为2维numpy array
        # print(vertices)
        # print(vertices.shape)

        # 转为axis aligned坐标
        vertices_coord=vertices[:,0:3]
        vertices_transpose=np.vstack([vertices_coord.T,np.ones([1,vertices_coord.shape[0]])])
        vertices_aligned=np.dot(axis_align_matrix,vertices_transpose)[0:3,:]
        vertices_aligned=vertices_aligned.T

        # 处理点云
        vertices_aligned=filter_pointcould(vertices_aligned)

        # 保存该物体的点云
        if save_per_instance_points:
            # draw_points(vertices_aligned)
            object_path=scannet_root+objects_info_folder_name+'/%s_%d_%s.npy'%(scan_id,obj_id,obj_label)
            np.save(object_path,vertices_aligned,allow_pickle=True)
            print("object points saved to:",object_path)


        # 计算相关参数并记录定量信息
        # x_max,x_min,x_avg=np.max(vertices[:,0]),np.min(vertices[:,0]),np.average(vertices[:,0])
        # y_max,y_min,y_avg=np.max(vertices[:,1]),np.min(vertices[:,1]),np.average(vertices[:,1])
        # z_max,z_min,z_avg=np.max(vertices[:,2]),np.min(vertices[:,2]),np.average(vertices[:,2])
        x_max,x_min,x_avg=np.max(vertices_aligned[:,0]),np.min(vertices_aligned[:,0]),np.average(vertices_aligned[:,0])
        y_max,y_min,y_avg=np.max(vertices_aligned[:,1]),np.min(vertices_aligned[:,1]),np.average(vertices_aligned[:,1])
        z_max,z_min,z_avg=np.max(vertices_aligned[:,2]),np.min(vertices_aligned[:,2]),np.average(vertices_aligned[:,2])

        # info["quan_info"]=[x_avg,y_avg,z_avg,(x_max-x_min),(y_max-y_min),(z_max-z_min)]
        info["centroid"]=[x_avg, y_avg, z_avg]
        info["center_position"]=[(x_max+x_min)/2, (y_max+y_min)/2, (z_max+z_min)/2]
        info["size"]=[(x_max-x_min),(y_max-y_min),(z_max-z_min)]
        info["extension"]=[x_min,y_min,z_min,x_max,y_max,z_max]
        info["median_rgba"]=[int(np.median(vertices[:,3])), int(np.median(vertices[:,4])), int(np.median(vertices[:,5])), int(np.median(vertices[:,6]))]

        # 读取oriented bbox 和 方向信息
        key=(scan_id,str(obj_id))
        # 如果在obb_data中能找到，则记录obb信息和旋转角度信息，并尝试记录front信息
        if key in obb_data:
            obb_info=obb_data[key]
            

            if (obb_info['catid_cad'], obb_info['id_cad']) in has_front_data:
                info['has_front']=True
                info['front_point']=obb_info['front_point']
            else:
                info['has_front']=False
                info['front_point']=obb_info['front_point']

            info['obb']=obb_info['obj_bbox']
            rot_matrix=np.array(obb_info['obj_rot']) 
            info['rot_angle']=np.arctan2(rot_matrix[1,0],rot_matrix[0,0]) #用x=atan2(sinx,cosx)。
            # rot 4x4太大了没必要全记录，测试了几个都是front_point算出来的角度+180

            # # for test
            # front_point=np.array(info['front_point'])
            # center_obb=np.array(info['obb'][0:3])
            # front_vector=front_point-center_obb
            # angle_calc=np.arctan2(front_vector[1],front_vector[0])
            # angle=info['rot_angle']
            # print(key)
            print("%s has front: %s" % (str(key),str(info['has_front'])))
            # print("angle from rot matrix:",angle/np.pi*180)
            # print("angle of front point:",angle_calc/np.pi*180)
            # print("center from referit3d:",center_obb)
            # print("center from ply:",info['center_position'])


        else:
            print("object not found in %s! %s"%(object_obb_data_path,str(key))) 
            info['has_front']=False
            info['front_point']=None
            info['obb']=None
            info['rot_angle']=None

        objects_info.append(info)

    

    save_path=scannet_root+"%s/%s_%s.npy"%(objects_info_folder_name,objects_info_folder_name,scan_id)
    np.save(save_path,objects_info)
    print("objects info save to:",save_path)

    return objects_info

def gen_obj_list_3dvista_align(scan_id,scannet_root,objects_info_folder_name,_3dvista_data_root=None,referit3d_data_root=None,conf_score_thr=0.5,save_per_instance_points=False,use_200_cats=True):
    # 各个文件的路径
    # aggregation_json_path = scannet_root+scan_id+"/"+scan_id+"_vh_clean.aggregation.json"
    # segs_json_path = scannet_root+scan_id+"/"+scan_id+"_vh_clean_2.0.010000.segs.json"
    ply_path = scannet_root+scan_id+"/"+scan_id+"_vh_clean_2.ply"
    object_obb_data_path=referit3d_data_root+'object_oriented_bboxes_aligned_scans.json'
    object_has_front_data_path=referit3d_data_root+'shapenet_has_front.csv'

    # scan_info_txt_path=mask3d_root+scan_id+'.txt'

    obj_mask_path = os.path.join(_3dvista_data_root, str(scan_id) + ".mask" + ".npz")
    obj_label_path = os.path.join(_3dvista_data_root, str(scan_id) + ".label" + ".npy")
    obj_score_path = os.path.join(_3dvista_data_root, str(scan_id) + ".score" + ".npy")
    obj_masks = np.array(sparse.load_npz(obj_mask_path).todense())[:50, :]
    obj_labels = np.load(obj_label_path)[:50]
    obj_scores = np.load(obj_score_path)[:50]
    # print(obj_scores)
    
    if not os.path.exists(scannet_root+objects_info_folder_name):
        os.mkdir(scannet_root+objects_info_folder_name)

    # # 打开txt
    # with open(scan_info_txt_path)as f:
    #     object_info_lines=f.readlines()


    # 读入ply文件
    print("loading .ply file...")
    plydata=PlyData.read(ply_path)

    header=plydata.header
    print("header of .ply file:",header)

    # 读入axis aline矩阵
    filename = scannet_root+scan_id+'/'+scan_id+'.txt'  # 替换成实际的文件名
    with open(filename, 'r') as file:
        lines=file.readlines()
        for line in lines:
            if "axisAlignment" in line:
                numbers = line.split('=')[1].strip().split()
                break
    axis_align_matrix = np.array(numbers, dtype=float).reshape(4, 4)

    # 读入obb和has_front文件
    obb_data=load_scan2cad_meta_data(object_obb_data_path)
    has_front_data=load_has_front_meta_data(object_has_front_data_path)

    # 定义最终返回的，记录所有object信息的dict
    objects_info=[]

    # 遍历 object masks
    for i,mask in enumerate(obj_masks):
        print("i:",i)
        obj_label_idx=obj_labels[i]
        obj_score=obj_scores[i]
        # print("obj_score:",obj_score)

        if use_200_cats:
            obj_label=idx_2_label_200(obj_label_idx)
        else:
            obj_label=idx_2_label_20(obj_label_idx)

        # 记录物体基本信息
        info={"id":i,
              "label":obj_label,
              "score":obj_score
              }
        
        # 在ply文件中找到物体对应的vertices
        vertices_all=np.array(plydata.elements[0].data)
        vertices=vertices_all[np.where(mask)]

        vertices=np.array([list(vertex) for vertex in vertices]) #转换为2维numpy array
        # print(vertices)
        # print(vertices.shape)

        # 转为axis aligned坐标
        vertices_coord=vertices[:,0:3]
        vertices_transpose=np.vstack([vertices_coord.T,np.ones([1,vertices_coord.shape[0]])])
        vertices_aligned=np.dot(axis_align_matrix,vertices_transpose)[0:3,:]
        vertices_aligned=vertices_aligned.T

        # 处理点云
        vertices_aligned=filter_pointcould(vertices_aligned)

        # 保存该物体的点云
        if save_per_instance_points:
            # draw_points(vertices_aligned)
            object_path=scannet_root+objects_info_folder_name+'/%s_%d_%s.npy'%(scan_id,i,obj_label)
            np.save(object_path,vertices_aligned,allow_pickle=True)
            print("object points saved to:",object_path)

        # 计算相关参数并记录定量信息
        # x_max,x_min,x_avg=np.max(vertices[:,0]),np.min(vertices[:,0]),np.average(vertices[:,0])
        # y_max,y_min,y_avg=np.max(vertices[:,1]),np.min(vertices[:,1]),np.average(vertices[:,1])
        # z_max,z_min,z_avg=np.max(vertices[:,2]),np.min(vertices[:,2]),np.average(vertices[:,2])
        x_max,x_min,x_avg=np.max(vertices_aligned[:,0]),np.min(vertices_aligned[:,0]),np.average(vertices_aligned[:,0])
        y_max,y_min,y_avg=np.max(vertices_aligned[:,1]),np.min(vertices_aligned[:,1]),np.average(vertices_aligned[:,1])
        z_max,z_min,z_avg=np.max(vertices_aligned[:,2]),np.min(vertices_aligned[:,2]),np.average(vertices_aligned[:,2])

        # info["quan_info"]=[x_avg,y_avg,z_avg,(x_max-x_min),(y_max-y_min),(z_max-z_min)]
        info["centroid"]=[x_avg, y_avg, z_avg]
        info["center_position"]=[(x_max+x_min)/2, (y_max+y_min)/2, (z_max+z_min)/2]
        info["size"]=[(x_max-x_min),(y_max-y_min),(z_max-z_min)]
        info["extension"]=[x_min,y_min,z_min,x_max,y_max,z_max]
        info["median_rgba"]=[int(np.median(vertices[:,3])), int(np.median(vertices[:,4])), int(np.median(vertices[:,5])), int(np.median(vertices[:,6]))]

        info['has_front']=False
        info['front_point']=None
        info['obb']=None
        info['rot_angle']=None

        objects_info.append(info)


    save_path=scannet_root+"%s/%s_%s.npy"%(objects_info_folder_name,objects_info_folder_name,scan_id)
    np.save(save_path,objects_info)
    print("objects info save to:",save_path)

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
    # # 处理train set
    # data_root="/share/data/ripl/scannet_raw/train/"
    # scan_id_list_train=get_scan_id_list(data_root)
    # referit3d_data_root="/share/data/ripl/vincenttann/sr3d/data/"
    # for idx,scan_id in enumerate(scan_id_list_train[1428:]):
    #     print("Processing train set: %s (%d/%d)..." % (scan_id,idx+1,len(scan_id_list_train)))
    #     objects_info=gen_obj_list(scan_id=scan_id, data_root=data_root,   referit3d_data_root=referit3d_data_root)
    #     np.save(data_root+"objects_info/objects_info_"+scan_id+".npy",objects_info,   allow_pickle=True)

    # # 处理test set
    # data_root="/share/data/ripl/scannet_raw/test/"
    # scan_id_list_test=get_scan_id_list(data_root)
    # for idx,scan_id in enumerate(scan_id_list_test):
    #     print("Processing test set: %s (%d/%d)..." % (scan_id,idx+1,len(scan_id_list_val)))
    #     objects_info=gen_obj_list(scan_id=scan_id, data_root=data_root,   referit3d_data_root=referit3d_data_root)
    #     np.save(data_root+"objects_info/objects_info_"+scan_id+".npy",objects_info,   allow_pickle=True)

    # ####### scanrefer数据，用group free的bbox #########
    # group_free_root="/share/data/ripl/vincenttann/sr3d/data/group_free_pred_bboxes/   group_free_pred_bboxes/"
    # record_file="/share/data/ripl/vincenttann/sr3d/record_gen_obj_list_gf.log"

    # # 处理train set
    # scannet_root="/share/data/ripl/scannet_raw/train/"
    # scan_id_list_train=get_scan_id_list(scannet_root)

    # for idx,scan_id in enumerate(scan_id_list_train):
    #     with open(record_file,'a') as f:
    #         f.write("Processing train set: %s (%d/%d)...\n" % (scan_id,idx+1,len  (scan_id_list_train)))
    #     print("Processing train set: %s (%d/%d)..." % (scan_id,idx+1,len(scan_id_list_train)))
    #     objects_info=gen_obj_list_group_free(scan_id=scan_id, 
    #                        scannet_root=scannet_root, 
    #                        group_free_root=group_free_root)
    #     np.save(scannet_root+"objects_info_gf/objects_info_gf_"+scan_id+".npy",objects_info,  allow_pickle=True)

    # # 处理test set 
    # # scannet的test set没给axis align矩阵
    # scannet_root="/share/data/ripl/scannet_raw/test/"
    # scan_id_list_test=get_scan_id_list(scannet_root)

    # for idx,scan_id in enumerate(scan_id_list_test):
    #     with open(record_file,'a') as f:
    #         f.write("Processing test set: %s (%d/%d)...\n" % (scan_id,idx+1,len   (scan_id_list_test)))
    #     print("Processing test set: %s (%d/%d)..." % (scan_id,idx+1,len(scan_id_list_test)))
    #     objects_info=gen_obj_list_group_free(scan_id=scan_id, 
    #                        scannet_root=scannet_root, 
    #                        group_free_root=group_free_root)
    #     np.save(scannet_root+"objects_info_gf/objects_info_gf_"+scan_id+".npy",objects_info,  allow_pickle=True)



    ## mask3d ##
    """use_200_cats=bool(1)
    if use_200_cats:
        mask3d_root="/share/data/ripl/vincenttann/Mask3D/eval_output/   instance_evaluation_mask3dTXS_export_scannet200_0/val/" # 200 cats
    else:
        mask3d_root="/share/data/ripl/vincenttann/Mask3D/eval_output/instance_evaluation_mask3dTXS_export_scannet_0/val_312/" # 20 cats

    referit3d_data_root="/share/data/ripl/vincenttann/sr3d/data/referit3d/"

    # mask3d处理val set 
    # scannet的test set没给axis align矩阵
    scannet_root="/share/data/ripl/scannet_raw/train/"
    # scan_id_list_test=get_scan_id_list(scannet_root)
    with open('data/scannetv2_val.txt') as f:
        scan_id_list_val=f.readlines()

    for idx,line in enumerate(scan_id_list_val):
        scan_id_list_val[idx]=line.strip()

    i=0
    start=50*i
    end=50*(i+1)
    max_idx=len(scan_id_list_val)-1
    if end<max_idx:
        scan_id_list_val_s=scan_id_list_val[start:end]
    else:
        scan_id_list_val_s=scan_id_list_val[start:]

    # scan_id_list_val_s=['scene0568_00']

    for idx,scan_id in enumerate(scan_id_list_val_s):
        # with open(record_file,'a') as f:
        #     f.write("Processing test set: %s (%d/%d)...\n" % (scan_id,idx+1,len   (scan_id_list_test)))
        print("Processing val set: %s (%d/%d)..." % (scan_id,idx+1,len(scan_id_list_val_s)))

        conf_score_thr=0.2
        # objects_info_folder_name="objects_info_mask3d_"+str(int(conf_score_thr*100))
        objects_info_folder_name="objects_info_mask3d_200c" if use_200_cats else "objects_info_mask3d_20c"
        objects_info=gen_obj_list_mask3d_align(scan_id=scan_id, 
                           scannet_root=scannet_root, 
                           objects_info_folder_name=objects_info_folder_name,
                           mask3d_root=mask3d_root,
                           referit3d_data_root=referit3d_data_root,
                           conf_score_thr=conf_score_thr,
                           save_per_instance_points=False,
                           use_200_cats=use_200_cats
                           )
        # np.save(scannet_root+"objects_info_mask3d_90/objects_info_mask3d_90_"+scan_id+".npy", objects_info,allow_pickle=True)"""


    ## 3d-vista ##
    use_200_cats=bool(1)
    _3dvista_data_root="/share/data/ripl/vincenttann/save_mask/save_mask/"
    referit3d_data_root="/share/data/ripl/vincenttann/sr3d/data/referit3d/"

    # mask3d处理val set 
    # scannet的test set没给axis align矩阵
    scannet_root="/share/data/ripl/scannet_raw/train/"
    # scan_id_list_test=get_scan_id_list(scannet_root)
    with open('data/scannetv2_val.txt') as f:
        scan_id_list_val=f.readlines()

    for idx,line in enumerate(scan_id_list_val):
        scan_id_list_val[idx]=line.strip()

    i=0
    start=50*i
    end=50*(i+1)
    max_idx=len(scan_id_list_val)-1
    if end<max_idx:
        scan_id_list_val_s=scan_id_list_val[start:end]
    else:
        scan_id_list_val_s=scan_id_list_val[start:]

    # scan_id_list_val_s=['scene0568_00']

    for idx,scan_id in enumerate(scan_id_list_val_s):
        # with open(record_file,'a') as f:
        #     f.write("Processing test set: %s (%d/%d)...\n" % (scan_id,idx+1,len   (scan_id_list_test)))
        print("Processing val set: %s (%d/%d)..." % (scan_id,idx+1,len(scan_id_list_val_s)))

        conf_score_thr=0.2
        # objects_info_folder_name="objects_info_mask3d_"+str(int(conf_score_thr*100))
        objects_info_folder_name="objects_info_3dvista_200c"
        objects_info=gen_obj_list_3dvista_align(scan_id=scan_id, 
                           scannet_root=scannet_root, 
                           objects_info_folder_name=objects_info_folder_name,
                           _3dvista_data_root=_3dvista_data_root,
                           referit3d_data_root=referit3d_data_root,
                           conf_score_thr=conf_score_thr,
                           save_per_instance_points=False,
                           use_200_cats=use_200_cats
                           )
        # np.save(scannet_root+"objects_info_mask3d_90/objects_info_mask3d_90_"+scan_id+".npy", objects_info,allow_pickle=True)
