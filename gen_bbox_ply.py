import numpy as np
import random
import os


def gen_bbox_ply_scanrefer(scannet_root,gf_root,scan_id,margin=0.01):
    npy_path=gf_root+scan_id+".npy"
    # npy_path="./data/group_free_pred_bboxes/group_free_pred_bboxes_train/scene0000_00.npy"
    boxes=np.load(npy_path,allow_pickle=True).reshape(-1)[0]['box']
    num_box=len(boxes)
    bounding_boxes=np.zeros([num_box,16,3])

    for idx,box in enumerate(boxes):
        xmin,ymin,zmin,xmax,ymax,zmax=box
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

    # 将边界框顶点坐标保存为一个 PLY 文件
    # out_file="H:\ScanNet Data\data\scannet\scans\scene0000_00\scene0000_00_bboxes_aligned.ply"
    out_file=scannet_root+scan_id+"/"+scan_id+"_bboxes_aligned.ply"
    with open(out_file, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(bounding_boxes) * 16}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")

        # f.write(f"element edge {len(bounding_boxes) * 12}\n")
        # # f.write(f"element edge 12\n")
        # f.write("property int vertex1\n")
        # f.write("property int vertex2\n")
        # # f.write("property uchar red\n")
        # # f.write("property uchar green\n")
        # # f.write("property uchar blue\n")

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
            # if i>0:
            #     continue
            # 随机选取颜色，避免出现白色（gt）
            color=[random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]
            min_idx=np.argmin(color)
            if color[min_idx]>220:
                color[min_idx]=0
            f.write("3 %d %d %d %d %d %d %d\n"%(16*i+0,  16*i+1,  16*i+3,  color[0],   color[1],   color[2],   i))
            f.write("3 %d %d %d %d %d %d %d\n"%(16*i+0,  16*i+3,  16*i+2,  color[0],   color[1],   color[2],   i))

            f.write("3 %d %d %d %d %d %d %d\n"%(16*i+2,  16*i+3,  16*i+5,  color[0],   color[1],   color[2],   i))
            f.write("3 %d %d %d %d %d %d %d\n"%(16*i+2,  16*i+5,  16*i+4,  color[0],   color[1],   color[2],   i))

            f.write("3 %d %d %d %d %d %d %d\n"%(16*i+4,  16*i+5,  16*i+7,  color[0],   color[1],   color[2],   i))
            f.write("3 %d %d %d %d %d %d %d\n"%(16*i+4,  16*i+7,  16*i+6,  color[0],   color[1],   color[2],   i))

            f.write("3 %d %d %d %d %d %d %d\n"%(16*i+0,  16*i+1,  16*i+7,  color[0],   color[1],   color[2],   i))
            f.write("3 %d %d %d %d %d %d %d\n"%(16*i+0,  16*i+7,  16*i+6,  color[0],   color[1],   color[2],   i))

            f.write("3 %d %d %d %d %d %d %d\n"%(16*i+0,  16*i+1,  16*i+9,  color[0],   color[1],   color[2],   i))
            f.write("3 %d %d %d %d %d %d %d\n"%(16*i+0,  16*i+9,  16*i+8,  color[0],   color[1],   color[2],   i))

            f.write("3 %d %d %d %d %d %d %d\n"%(16*i+2,  16*i+3,  16*i+11, color[0],   color[1],   color[2],   i))
            f.write("3 %d %d %d %d %d %d %d\n"%(16*i+2,  16*i+11, 16*i+10, color[0],   color[1],   color[2],   i))

            f.write("3 %d %d %d %d %d %d %d\n"%(16*i+4,  16*i+5,  16*i+13, color[0],   color[1],   color[2],   i))
            f.write("3 %d %d %d %d %d %d %d\n"%(16*i+4,  16*i+13, 16*i+12, color[0],   color[1],   color[2],   i))

            f.write("3 %d %d %d %d %d %d %d\n"%(16*i+6,  16*i+7,  16*i+15, color[0],   color[1],   color[2],   i))
            f.write("3 %d %d %d %d %d %d %d\n"%(16*i+6,  16*i+15, 16*i+14, color[0],   color[1],   color[2],   i))

            f.write("3 %d %d %d %d %d %d %d\n"%(16*i+8,  16*i+9,  16*i+11, color[0],   color[1],   color[2],   i))
            f.write("3 %d %d %d %d %d %d %d\n"%(16*i+8,  16*i+11, 16*i+10, color[0],   color[1],   color[2],   i))

            f.write("3 %d %d %d %d %d %d %d\n"%(16*i+10, 16*i+11, 16*i+13, color[0],   color[1],   color[2],   i))
            f.write("3 %d %d %d %d %d %d %d\n"%(16*i+10, 16*i+13, 16*i+12, color[0],   color[1],   color[2],   i))

            f.write("3 %d %d %d %d %d %d %d\n"%(16*i+12, 16*i+13, 16*i+15, color[0],   color[1],   color[2],   i))
            f.write("3 %d %d %d %d %d %d %d\n"%(16*i+12, 16*i+15, 16*i+14, color[0],   color[1],   color[2],   i))

            f.write("3 %d %d %d %d %d %d %d\n"%(16*i+8,  16*i+9,  16*i+15, color[0],   color[1],   color[2],   i))
            f.write("3 %d %d %d %d %d %d %d\n"%(16*i+8,  16*i+15, 16*i+14, color[0],   color[1],   color[2],   i))

    print("Bounding boxes data saved to %s"%out_file)

def gen_bbox_ply_mask3d(scannet_root,scan_id,suffix,margin=0.01,thr=0.2):
    # 该函数需要已经保存好的object_info
    # npy_path="/share/data/ripl/scannet_raw/train/objects_info_mask3d/objects_info_mask3d_%s.npy"%scan_id
    # npy_path="%sobjects_info_mask3d/objects_info_mask3d_%s.npy"%(scannet_root,scan_id)
    # npy_path="%sobjects_info_mask3d_35/objects_info_mask3d_35_%s.npy"%(scannet_root,scan_id)
    npy_path="%sobjects_info_mask3d_%s/objects_info_mask3d_%s_%s.npy"%(scannet_root,suffix,suffix,scan_id)
    object_info=np.load(npy_path,allow_pickle=True)
    # boxes=np.load(npy_path,allow_pickle=True).reshape(-1)[0]['box']
    num_box=len(object_info)
    bounding_boxes=np.zeros([num_box,16,3])

    total_count=0
    confident_count=0

    # 遍历boxes，把顶点信息（每个box有8*2=16个顶点）保存到bounding_boxes变量
    for idx,obj in enumerate(object_info):
        total_count+=1
        if obj['score']<thr:
            continue
        confident_count+=1
        ctr=obj['center_position']
        size=obj['size']
        xmin,xmax=ctr[0]-size[0]/2,ctr[0]+size[0]/2
        ymin,ymax=ctr[1]-size[1]/2,ctr[1]+size[1]/2
        zmin,zmax=ctr[2]-size[2]/2,ctr[2]+size[2]/2
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
        
    print("total:",total_count)
    print("above confidential thr:",confident_count)

    # 将边界框顶点坐标保存为一个 PLY 文件
    # out_file="H:\ScanNet Data\data\scannet\scans\scene0000_00\scene0000_00_bboxes_aligned.ply"
    out_file=scannet_root+scan_id+"/"+scan_id+"_bboxes_aligned_mask3d_%s_%s.ply"%(suffix,str(int(100*thr)))
    with open(out_file, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(bounding_boxes) * 16}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")

        # f.write(f"element edge {len(bounding_boxes) * 12}\n")
        # # f.write(f"element edge 12\n")
        # f.write("property int vertex1\n")
        # f.write("property int vertex2\n")
        # # f.write("property uchar red\n")
        # # f.write("property uchar green\n")
        # # f.write("property uchar blue\n")

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
            # if i>0:
            #     continue
            # 随机选取颜色，避免出现白色（gt）
            color=[random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]
            min_idx=np.argmin(color)
            if color[min_idx]>220:
                color[min_idx]=0
            f.write("3 %d %d %d %d %d %d %d\n"%(16*i+0,  16*i+1,  16*i+3,  color[0],   color[1],   color[2],   i))
            f.write("3 %d %d %d %d %d %d %d\n"%(16*i+0,  16*i+3,  16*i+2,  color[0],   color[1],   color[2],   i))

            f.write("3 %d %d %d %d %d %d %d\n"%(16*i+2,  16*i+3,  16*i+5,  color[0],   color[1],   color[2],   i))
            f.write("3 %d %d %d %d %d %d %d\n"%(16*i+2,  16*i+5,  16*i+4,  color[0],   color[1],   color[2],   i))

            f.write("3 %d %d %d %d %d %d %d\n"%(16*i+4,  16*i+5,  16*i+7,  color[0],   color[1],   color[2],   i))
            f.write("3 %d %d %d %d %d %d %d\n"%(16*i+4,  16*i+7,  16*i+6,  color[0],   color[1],   color[2],   i))

            f.write("3 %d %d %d %d %d %d %d\n"%(16*i+0,  16*i+1,  16*i+7,  color[0],   color[1],   color[2],   i))
            f.write("3 %d %d %d %d %d %d %d\n"%(16*i+0,  16*i+7,  16*i+6,  color[0],   color[1],   color[2],   i))

            f.write("3 %d %d %d %d %d %d %d\n"%(16*i+0,  16*i+1,  16*i+9,  color[0],   color[1],   color[2],   i))
            f.write("3 %d %d %d %d %d %d %d\n"%(16*i+0,  16*i+9,  16*i+8,  color[0],   color[1],   color[2],   i))

            f.write("3 %d %d %d %d %d %d %d\n"%(16*i+2,  16*i+3,  16*i+11, color[0],   color[1],   color[2],   i))
            f.write("3 %d %d %d %d %d %d %d\n"%(16*i+2,  16*i+11, 16*i+10, color[0],   color[1],   color[2],   i))

            f.write("3 %d %d %d %d %d %d %d\n"%(16*i+4,  16*i+5,  16*i+13, color[0],   color[1],   color[2],   i))
            f.write("3 %d %d %d %d %d %d %d\n"%(16*i+4,  16*i+13, 16*i+12, color[0],   color[1],   color[2],   i))

            f.write("3 %d %d %d %d %d %d %d\n"%(16*i+6,  16*i+7,  16*i+15, color[0],   color[1],   color[2],   i))
            f.write("3 %d %d %d %d %d %d %d\n"%(16*i+6,  16*i+15, 16*i+14, color[0],   color[1],   color[2],   i))

            f.write("3 %d %d %d %d %d %d %d\n"%(16*i+8,  16*i+9,  16*i+11, color[0],   color[1],   color[2],   i))
            f.write("3 %d %d %d %d %d %d %d\n"%(16*i+8,  16*i+11, 16*i+10, color[0],   color[1],   color[2],   i))

            f.write("3 %d %d %d %d %d %d %d\n"%(16*i+10, 16*i+11, 16*i+13, color[0],   color[1],   color[2],   i))
            f.write("3 %d %d %d %d %d %d %d\n"%(16*i+10, 16*i+13, 16*i+12, color[0],   color[1],   color[2],   i))

            f.write("3 %d %d %d %d %d %d %d\n"%(16*i+12, 16*i+13, 16*i+15, color[0],   color[1],   color[2],   i))
            f.write("3 %d %d %d %d %d %d %d\n"%(16*i+12, 16*i+15, 16*i+14, color[0],   color[1],   color[2],   i))

            f.write("3 %d %d %d %d %d %d %d\n"%(16*i+8,  16*i+9,  16*i+15, color[0],   color[1],   color[2],   i))
            f.write("3 %d %d %d %d %d %d %d\n"%(16*i+8,  16*i+15, 16*i+14, color[0],   color[1],   color[2],   i))

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

gf_root="H:/CodeUndergrad/Refer3dProject/Sr3d/data/group_free_pred_bboxes/group_free_pred_bboxes/"
scannet_root="H:/ScanNet_Data/data/scannet/scans/"

# scan_id_list=get_scan_id_list(scannet_root)
# scan_id_list=["scene0000_00",]

# for scan_id in scan_id_list:
#     gen_bbox_ply_scanrefer(scannet_root,gf_root,scan_id,margin=0.02)

val_set_txt="data/scannetv2_val.txt"
with open(val_set_txt) as f:
    scan_id_list=f.readlines()
for idx,scan_id in enumerate(scan_id_list):
    scan_id_list[idx]=scan_id.strip()

# scan_id_list=['scene0568_00']

for scan_id in scan_id_list:
    gen_bbox_ply_mask3d(scannet_root,scan_id,suffix='200c',margin=0.02,thr=0.4)

