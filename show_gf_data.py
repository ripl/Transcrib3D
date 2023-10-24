import numpy as np
path="./data/group_free_pred_bboxes/group_free_pred_bboxes_train/scene0000_00.npy"
data=np.load(path,allow_pickle=True).reshape(-1)[0]
print(data['class'][0])
print(data['box'][0]) 
print(data['pc'].shape)