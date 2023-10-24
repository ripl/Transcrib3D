import numpy as np
import random
from plyfile import PlyData, PlyElement

mask3d_root="H:/CodeUndergrad/Refer3dProject/instance_evaluation_mask3dTXS_export_scannet200_0/val/"
scan_id="scene0568_00"
# scan_info_txt_path="H:\CodeUndergrad\Refer3dProject\instance_evaluation_mask3dTXS_export_scannet200_0\val\scene0568_00.txt"
scan_info_txt_path=mask3d_root+scan_id+'.txt'

# 打开txt
with open(scan_info_txt_path)as f:
    object_info_lines=f.readlines()

mask_arrays=[]
num_objects=0

# 遍历objects
for line in object_info_lines:
    line_split=line.split()
    print(line_split)
    obj_mask_path=line_split[0]
    score=eval(line_split[2])
    if score<0.9:
        continue
    obj_id=int(obj_mask_path.split('.')[0].split('_')[-1])
    # obj_label_idx=int(line_split[1])
    # obj_label=idx_2_label(obj_label_idx)
    with open(mask3d_root+obj_mask_path) as f:
        mask_lines=f.readlines()
        mask_data= [int(mask_line.strip()) for mask_line in mask_lines]
        print(np.sum(mask_data))
        mask_arrays.append(mask_data)
        num_objects+=1

# 读取原始PLY文件
original_ply = PlyData.read('H:\ScanNet_Data\data\scannet\scans\scene0568_00\scene0568_00_vh_clean_2_aligned.ply')

# 获取原始顶点数据和面数据
vertices = np.vstack([original_ply['vertex']['x'],
                      original_ply['vertex']['y'],
                      original_ply['vertex']['z'],
                      original_ply['vertex']['red'],
                      original_ply['vertex']['green'],
                      original_ply['vertex']['blue'],
                      original_ply['vertex']['alpha'],
                      ]).T
print(vertices.shape)
faces = original_ply['face']['vertex_indices']

# 假设有多个物体的mask数组，每个数组的长度与顶点数相同
# 这里假设有三个物体的mask数组
# num_objects = 3
# mask_arrays = []

# 为每个物体生成随机RGB颜色
object_colors = [tuple(random.randint(0, 255) for _ in range(3)) for _ in range(num_objects)]

# 初始化一个新的顶点颜色数组
vertex_colors = np.zeros((len(vertices), 3), dtype=np.uint8)

# 遍历每个物体的mask数组
for object_id, mask_array in enumerate(mask_arrays):
    # 将属于当前物体的顶点索引找出来
    # print(mask_array)
    object_vertices = np.where(np.array(mask_array) == 1)
    # print(object_vertices)

    # 为这些顶点设置相同的RGB颜色
    vertex_colors[object_vertices] = object_colors[object_id]
    # print(object_colors[object_id])

# 创建一个新的PLY文件
vertex_properties = [('x', 'f4'), ('y', 'f4'), ('z', 'f4')]
# new_ply = PlyData([PlyElement.describe(vertices, 'vertex',property=vertex_properties),
#                    PlyElement.describe(faces, 'face')])
from copy import deepcopy
new_ply=deepcopy(original_ply)

# 将新的顶点颜色数据写入PLY文件
new_ply['vertex']['red'] = vertex_colors[:, 0]
new_ply['vertex']['green'] = vertex_colors[:, 1]
new_ply['vertex']['blue'] = vertex_colors[:, 2]
new_ply['vertex']['alpha'] = np.arange(len(vertices))

# 保存新的PLY文件
new_ply.write('H:\ScanNet_Data\data\scannet\scans\scene0568_00\scene0568_00_vh_clean_2_aligned_seg_mask3d_90.ply')
