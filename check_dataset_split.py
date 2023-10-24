import json

# 获取ScanNet原始的train、val和test
with open('data/scannetv2_train.txt') as f:
    scans=f.readlines()
    for idx,scan in enumerate(scans):
        scans[idx]=scan.strip()
    scannet_train=set(scans)
with open('data/scannetv2_val.txt') as f:
    scans=f.readlines()
    for idx,scan in enumerate(scans):
        scans[idx]=scan.strip()
    scannet_val=set(scans)
# with open('data/scannetv2_test.txt') as f:
#     scans=f.readlines()
#     for idx,scan in enumerate(scans):
#         scans[idx]=scan.strip()
#     scannet_test=set(scans)

scanrefer_train=set()
with open('data/scanrefer/ScanRefer_filtered_train.json') as f:
    data=json.load(f)
    for d in data:
        scanrefer_train.add(d['scene_id'])

scanrefer_val=set()
with open('data/scanrefer/ScanRefer_filtered_val.json') as f:
    data=json.load(f)
    for d in data:
        scanrefer_val.add(d['scene_id'])

print("train:")
print(len(scannet_train),len(scanrefer_train))
print("equal:",scannet_train==scanrefer_train)
print("subset:",scanrefer_train.issubset(scannet_train))

print("val:")
print(len(scannet_val),len(scanrefer_val))
print("equal:",scannet_val==scanrefer_val)
print("subset:",scanrefer_val.issubset(scannet_val))