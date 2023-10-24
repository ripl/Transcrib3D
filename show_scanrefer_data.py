# 本文件用于显示scanrefer特定编号的数据
import json

def read_json(file_path):
    # 本函数用于读入scanrefer数据(.json)，返回的格式和原json相同，对于scanrefer来说就是字典的list，索引从0开始
    with open(file_path, 'r') as jf:
        jf_data=jf.read() #这里得到jf_data是string类型
        data=json.loads(jf_data) #得到了和json文件一样的结构:data是dict的list
    return data

path="./data/scanrefer/ScanRefer_filtered_sampled50.json"
scanrefer_data=read_json(path)

idx=46

data=scanrefer_data[idx]
print("scan_id:%s"%data['scene_id'])
print("utterance:%s"%data['description'])
print("target id:%s"%data['object_id'])