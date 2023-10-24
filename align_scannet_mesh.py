# from plyfile import PlyData,PlyElement
# from copy import deepcopy

# path="H:\ScanNet Data\data\scannet\scans\scene0000_00\scene0000_00_vh_clean_2.ply"
# path_al="H:\ScanNet Data\data\scannet\scans\scene0000_00\scene0000_00_vh_clean_2_aligned.ply"

# plydata=PlyData.read(path)
# print(plydata.header)

# plydata_al=PlyData.read(path_al)
# print(plydata_al.header)


# vertex=plydata.elements[0]
# print(vertex[0])

# vertex_al=plydata_al.elements[0]
# print(vertex_al[0])

# face=plydata.elements[1]
# print(face[0])

# # print(vertex['x'])

# print(type(vertex))
# print(type(plydata['vertex']))
# print(dir(vertex))
# print(vertex.__len__())


import numpy as np
from plyfile import PlyData, PlyElement
from copy import deepcopy
import os

# 读取原始PLY文件
def read_ply(file_path):
    plydata = PlyData.read(file_path)
    return plydata

# 坐标变换函数
def transform_vertices(vertices, transformation_matrix):
    homogeneous_coords = np.hstack((np.array(vertices['x']).reshape(-1,1), np.array(vertices['y']).reshape(-1,1), np.array(vertices['z']).reshape(-1,1), np.ones(len(vertices)).reshape(-1,1)))
    transformed_coords = np.dot(transformation_matrix, homogeneous_coords.T).T
    transformed_vertices = deepcopy(vertices)
    transformed_vertices['x'] = transformed_coords[:, 0]
    transformed_vertices['y'] = transformed_coords[:, 1]
    transformed_vertices['z'] = transformed_coords[:, 2]
    return transformed_vertices

# 保存PLY文件
def save_ply(file_path, vertices, faces):
    num_vertices = len(vertices)
    num_faces = len(faces)

    vertex_element = PlyElement.describe(np.array(vertices), 'vertex')
    face_element = PlyElement.describe(np.array(faces), 'face')

    PlyData([vertex_element, face_element]).write(file_path)

# 读取原始PLY文件
# scannet_root= "/share/data/ripl/scannet_raw/train/"
scannet_root= "H:/ScanNet Data/data/scannet/scans/"

# 获取所有子文件夹的名称
scan_ids = [f for f in os.listdir(scannet_root) if os.path.isdir(os.path.join(scannet_root, f)) and 'scene' in f]

print("scan_ids:", scan_ids)

# record_file="/share/data/ripl/vincenttann/sr3d/record_align_mesh.log"
record_file="./record_align_mesh.log"
with open(record_file,'a') as f:
    f.write(str(scan_ids))
    f.write('\n')

for idx,scan_id in enumerate(scan_ids):
    print("Processing %s, %d/%d"%(scan_id,idx+1,len(scan_ids)))
    with open(record_file,'a') as f:
        f.write("Processing %s, %d/%d\n"%(scan_id,idx+1,len(scan_ids)))

    # input_ply = "H:\ScanNet Data\data\scannet\scans\scene0000_00\scene0000_00_vh_clean_2.ply"
    # output_ply = "H:\ScanNet Data\data\scannet\scans\scene0000_00\scene0000_00_vh_clean_2_aligned.ply"
    
    input_ply=scannet_root+scan_id+'/'+scan_id+"_vh_clean_2.ply"
    output_ply=scannet_root+scan_id+'/'+scan_id+"_vh_clean_2_aligned.ply"

    plydata = read_ply(input_ply)

    # 获取顶点和面数据
    vertices = plydata['vertex']
    # vertices=np.array([list(vertex) for vertex in vertices]) #转换为2维numpy array
    faces = plydata['face']

    # 读入axis aline矩阵
    filename = scannet_root+scan_id+'/'+scan_id+".txt"
    with open(filename, 'r') as file:
        lines=file.readlines()
        for line in lines:
            if "axisAlignment" in line:
                numbers = line.split('=')[1].strip().split()
                break
    axis_align_matrix = np.array(numbers, dtype=float).reshape(4, 4)

    # 进行坐标变换
    transformed_vertices = transform_vertices(vertices, axis_align_matrix)

    # 保存坐标变换后的PLY文件
    save_ply(output_ply, transformed_vertices, faces)

    print("Transformation completed and saved to", output_ply)
