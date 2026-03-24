import numpy as np
import random
from plyfile import PlyData, PlyElement

mask3d_root="H:/CodeUndergrad/Refer3dProject/instance_evaluation_mask3dTXS_export_scannet200_0/val/"
scan_id="scene0568_00"
# scan_info_txt_path="H:\CodeUndergrad\Refer3dProject\instance_evaluation_mask3dTXS_export_scannet200_0\val\scene0568_00.txt"
scan_info_txt_path=mask3d_root+scan_id+'.txt'

# Open the txt file
with open(scan_info_txt_path)as f:
    object_info_lines=f.readlines()

mask_arrays=[]
num_objects=0

# Iterate through the objects
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

# Read the original PLY file
original_ply = PlyData.read('H:\ScanNet_Data\data\scannet\scans\scene0568_00\scene0568_00_vh_clean_2_aligned.ply')

# Get the original vertex and face data
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

# Assume there are multiple object mask arrays, each with the same length as the number of vertices
# Assume there are three object mask arrays here
# num_objects = 3
# mask_arrays = []

# Generate a random RGB color for each object
object_colors = [tuple(random.randint(0, 255) for _ in range(3)) for _ in range(num_objects)]

# Initialize a new vertex color array
vertex_colors = np.zeros((len(vertices), 3), dtype=np.uint8)

# Iterate through each object's mask array
for object_id, mask_array in enumerate(mask_arrays):
    # Find the vertex indices that belong to the current object
    # print(mask_array)
    object_vertices = np.where(np.array(mask_array) == 1)
    # print(object_vertices)

    # Assign the same RGB color to these vertices
    vertex_colors[object_vertices] = object_colors[object_id]
    # print(object_colors[object_id])

# Create a new PLY file
vertex_properties = [('x', 'f4'), ('y', 'f4'), ('z', 'f4')]
# new_ply = PlyData([PlyElement.describe(vertices, 'vertex',property=vertex_properties),
#                    PlyElement.describe(faces, 'face')])
from copy import deepcopy
new_ply=deepcopy(original_ply)

# Write the new vertex color data into the PLY file
new_ply['vertex']['red'] = vertex_colors[:, 0]
new_ply['vertex']['green'] = vertex_colors[:, 1]
new_ply['vertex']['blue'] = vertex_colors[:, 2]
new_ply['vertex']['alpha'] = np.arange(len(vertices))

# Save the new PLY file
new_ply.write('H:\ScanNet_Data\data\scannet\scans\scene0568_00\scene0568_00_vh_clean_2_aligned_seg_mask3d_90.ply')
