import numpy as np
from plyfile import PlyData, PlyElement
from copy import deepcopy
import os
import argparse

def read_ply(file_path):
    plydata = PlyData.read(file_path)
    return plydata

def transform_vertices(vertices, transformation_matrix):
    homogeneous_coords = np.hstack((np.array(vertices['x']).reshape(-1,1), np.array(vertices['y']).reshape(-1,1), np.array(vertices['z']).reshape(-1,1), np.ones(len(vertices)).reshape(-1,1)))
    transformed_coords = np.dot(transformation_matrix, homogeneous_coords.T).T
    transformed_vertices = deepcopy(vertices)
    transformed_vertices['x'] = transformed_coords[:, 0]
    transformed_vertices['y'] = transformed_coords[:, 1]
    transformed_vertices['z'] = transformed_coords[:, 2]
    return transformed_vertices

def save_ply(file_path, vertices, faces):
    vertex_element = PlyElement.describe(np.array(vertices), 'vertex')
    face_element = PlyElement.describe(np.array(faces), 'face')

    PlyData([vertex_element, face_element]).write(file_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Align ScanNet Meshes")
    parser.add_argument("--scannet_download_path", type=str, help="Path of the ScanNet data download folder. There should be subfolder 'scans' under it.")
    args = parser.parse_args()
    scannet_download_path = args.scannet_download_path
    scans_path = os.path.join(scannet_download_path, 'scans')

    # names of all subfolders
    scan_ids = [f for f in os.listdir(scans_path) if os.path.isdir(os.path.join(scans_path, f)) and 'scene' in f]

    print("scan_ids:", scan_ids)

    # # additional logging
    # record_file="./record_align_mesh.log"
    # with open(record_file,'a') as f:
    #     f.write(str(scan_ids))
    #     f.write('\n')

    for idx,scan_id in enumerate(scan_ids):
        print("Processing %s, %d/%d"%(scan_id,idx+1,len(scan_ids)))
        # with open(record_file,'a') as f:
        #     f.write("Processing %s, %d/%d\n"%(scan_id,idx+1,len(scan_ids)))

        input_ply = os.path.join(scans_path, scan_id, scan_id+"_vh_clean_2.ply")
        output_ply = os.path.join(scans_path, scan_id, scan_id+"_vh_clean_2_aligned.ply")

        plydata = read_ply(input_ply)

        # vertices and faces
        vertices = plydata['vertex']
        faces = plydata['face']

        # read in axis align matrix
        filename = os.path.join(scans_path, scan_id, scan_id+".txt")
        with open(filename, 'r') as file:
            lines=file.readlines()
            for line in lines:
                if "axisAlignment" in line:
                    numbers = line.split('=')[1].strip().split()
                    break
        axis_align_matrix = np.array(numbers, dtype=float).reshape(4, 4)

        # transform and save
        transformed_vertices = transform_vertices(vertices, axis_align_matrix)

        save_ply(output_ply, transformed_vertices, faces)
        print("Transformation completed and saved to", output_ply)
