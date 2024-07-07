import numpy as np
import random
import os

def gen_gt_bbox_ply(scannet_root,scanrefer_gt_root,scan_id,margin=0.01):
    gt_box_npy_path=scanrefer_gt_root+scan_id+"_aligned_bbox.npy"
    boxes=np.load(gt_box_npy_path,allow_pickle=True)
    num_box=len(boxes)

    bounding_boxes=np.zeros([num_box,16,3])

    for idx,box in enumerate(boxes):
        cx,cy,cz,sx,sy,sz=box[0:6]
        xmin=cx-sx/2
        xmax=cx+sx/2
        ymin=cy-sy/2
        ymax=cy+sy/2
        zmin=cz-sz/2
        zmax=cz+sz/2
        bounding_boxes[idx]=np.array(
            [[xmin, ymin, zmin], #0
             [xmin-margin, ymin-margin, zmin], #1
             [xmax, ymin, zmin], #2
             [xmax+margin, ymin-margin, zmin], #3
             [xmax, ymax, zmin], #4
             [xmax+margin, ymax+margin, zmin], #5
             [xmin, ymax, zmin], #6
             [xmin-margin, ymax+margin, zmin], #7
             [xmin, ymin, zmax], #8
             [xmin-margin, ymin-margin, zmax], #9
             [xmax, ymin, zmax], #10
             [xmax+margin, ymin-margin, zmax], #11
             [xmax, ymax, zmax], #12
             [xmax+margin, ymax+margin, zmax], #13
             [xmin, ymax, zmax], #14
             [xmin-margin, ymax+margin, zmax] #15
            ])
        # print(box)

    # 将边界框顶点坐标保存为一个 PLY 文件
    # out_file="H:\ScanNet Data\data\scannet\scans\scene0000_00\scene0000_00_bboxes_aligned.ply"
    out_file=scannet_root+scan_id+"/"+scan_id+"_gt_bboxes_aligned.ply"
    with open(out_file, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(bounding_boxes) * 16}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")

        f.write(f"element face {len(bounding_boxes) * 24}\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("property uchar alpha\n")
        f.write("end_header\n")
        f.write("\n")

        # 具体数据
        # vertex坐标
        for bbox in bounding_boxes:
            # color=(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            for vertex in bbox:
                # f.write(f"{vertex[0]} {vertex[1]} {vertex[2]} {color[0]} {color[1]} {color[2]}\n")
                f.write(f"{vertex[0]} {vertex[1]} {vertex[2]}\n")
        f.write("\n")
        # face
        for i in range(len(bounding_boxes)):
            # color=(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            color=[255,255,255]
            object_id=boxes[i,-1]
            f.write("3 %d %d %d %d %d %d %d\n"%(16*i+0,  16*i+1,  16*i+3,  color[0],   color[1],   color[2],   i))
            f.write("3 %d %d %d %d %d %d %d\n"%(16*i+0,  16*i+3,  16*i+2,  color[0],   color[1],   color[2],   i))

            f.write("3 %d %d %d %d %d %d %d\n"%(16*i+2,  16*i+3,  16*i+5,  color[0],   color[1],   color[2],   object_id))
            f.write("3 %d %d %d %d %d %d %d\n"%(16*i+2,  16*i+5,  16*i+4,  color[0],   color[1],   color[2],   object_id))

            f.write("3 %d %d %d %d %d %d %d\n"%(16*i+4,  16*i+5,  16*i+7,  color[0],   color[1],   color[2],   object_id))
            f.write("3 %d %d %d %d %d %d %d\n"%(16*i+4,  16*i+7,  16*i+6,  color[0],   color[1],   color[2],   object_id))

            f.write("3 %d %d %d %d %d %d %d\n"%(16*i+0,  16*i+1,  16*i+7,  color[0],   color[1],   color[2],   object_id))
            f.write("3 %d %d %d %d %d %d %d\n"%(16*i+0,  16*i+7,  16*i+6,  color[0],   color[1],   color[2],   object_id))

            f.write("3 %d %d %d %d %d %d %d\n"%(16*i+0,  16*i+1,  16*i+9,  color[0],   color[1],   color[2],   object_id))
            f.write("3 %d %d %d %d %d %d %d\n"%(16*i+0,  16*i+9,  16*i+8,  color[0],   color[1],   color[2],   object_id))

            f.write("3 %d %d %d %d %d %d %d\n"%(16*i+2,  16*i+3,  16*i+11, color[0],   color[1],   color[2],   object_id))
            f.write("3 %d %d %d %d %d %d %d\n"%(16*i+2,  16*i+11, 16*i+10, color[0],   color[1],   color[2],   object_id))

            f.write("3 %d %d %d %d %d %d %d\n"%(16*i+4,  16*i+5,  16*i+13, color[0],   color[1],   color[2],   object_id))
            f.write("3 %d %d %d %d %d %d %d\n"%(16*i+4,  16*i+13, 16*i+12, color[0],   color[1],   color[2],   object_id))

            f.write("3 %d %d %d %d %d %d %d\n"%(16*i+6,  16*i+7,  16*i+15, color[0],   color[1],   color[2],   object_id))
            f.write("3 %d %d %d %d %d %d %d\n"%(16*i+6,  16*i+15, 16*i+14, color[0],   color[1],   color[2],   object_id))

            f.write("3 %d %d %d %d %d %d %d\n"%(16*i+8,  16*i+9,  16*i+11, color[0],   color[1],   color[2],   object_id))
            f.write("3 %d %d %d %d %d %d %d\n"%(16*i+8,  16*i+11, 16*i+10, color[0],   color[1],   color[2],   object_id))

            f.write("3 %d %d %d %d %d %d %d\n"%(16*i+10, 16*i+11, 16*i+13, color[0],   color[1],   color[2],   object_id))
            f.write("3 %d %d %d %d %d %d %d\n"%(16*i+10, 16*i+13, 16*i+12, color[0],   color[1],   color[2],   object_id))

            f.write("3 %d %d %d %d %d %d %d\n"%(16*i+12, 16*i+13, 16*i+15, color[0],   color[1],   color[2],   object_id))
            f.write("3 %d %d %d %d %d %d %d\n"%(16*i+12, 16*i+15, 16*i+14, color[0],   color[1],   color[2],   object_id))

            f.write("3 %d %d %d %d %d %d %d\n"%(16*i+8,  16*i+9,  16*i+15, color[0],   color[1],   color[2],   object_id))
            f.write("3 %d %d %d %d %d %d %d\n"%(16*i+8,  16*i+15, 16*i+14, color[0],   color[1],   color[2],   object_id))

    print("Bounding boxes data saved to %s"%out_file)

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


scanrefer_gt_root="H:/CodeUndergrad/Refer3dProject/ScanRefer/data/scannet/scannet_data/"
gf_root="H:/CodeUndergrad/Refer3dProject/Sr3d/data/group_free_pred_bboxes/group_free_pred_bboxes/"
scannet_root="H:/ScanNet Data/data/scannet/scans/"

data_list=get_scan_id_list(scannet_root)
# data_list=["scene0000_00",]
for scan_id in data_list:
    gen_gt_bbox_ply(scannet_root,scanrefer_gt_root,scan_id,margin=0.02)

