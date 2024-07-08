from tools.draw_obj_points import draw_points
import os
import numpy as np
from sklearn.cluster import DBSCAN
from collections import Counter

dbscan=DBSCAN(eps=0.1,min_samples=20)

objects_info_folder_path="H:\ScanNet_Data\data\scannet\scans\objects_info_mask3d_90"
files=os.listdir(objects_info_folder_path)

for file in files:
    if file.startswith("scene"):
        print(file)
        points=np.load(os.path.join(objects_info_folder_path,file),allow_pickle=True)
        dbscan.fit(points)
        # print(dbscan.labels_)     
        counter=Counter(dbscan.labels_)
        main_idx=counter.most_common()[0][0]      
        print("counter:",counter)
        print("main_idx:",main_idx)
        # main_idx=max(counter,counter.get)
        points_filtered=points[dbscan.labels_==main_idx]
        # print(main_idx)
        draw_points(points,title=file)
        draw_points(points_filtered,title=file+' filtered')