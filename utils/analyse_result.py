import numpy as np
import os

from utils.read_data import load_refer_dataset_pure

def get_easy_info(dataset_type, refer_dataset, line_number) -> bool:
    # 对于sr3d和nr3d，检查line_number对应数据的难度。
    # 同类物体（包括正确物体自身）<=2个则为easy，否则为hard
    if dataset_type == 'sr3d':
        refer_data = refer_dataset[int(line_number)]
        distractor_ids = eval(refer_data['distractor_ids'])
        # print(distractor_ids)
        is_easy = True if len(distractor_ids) <= 1 else False
    else:
        refer_data = refer_dataset[int(line_number)]
        # nr3d的stimulus_id格式: scan_id-target_class-target_id-distractor_id1-...-distractor_idn
        stimulus_id = refer_data['stimulus_id']
        n_object_same_class = int(stimulus_id.split('-')[2])
        is_easy = True if n_object_same_class <= 2 else False
    return is_easy

def get_view_dep_info(dataset_type, refer_dataset, line_number, use_original_viewdep_judge=True) -> bool:
    # 对于sr3d和nr3d，检查utterance是否为view_dependent.对照了referit3d和butd的代码，只需要检查utterance中是否出现以下关键词
    refer_data = refer_dataset[int(line_number)]
    utterance = refer_data['utterance']
    rels = [
        'front', 'behind', 'back', 'left', 'right', 'facing',
        'leftmost', 'rightmost', 'looking', 'across'
    ]
    if use_original_viewdep_judge:
        words = set(utterance.split())  # ... on the left.
        return any(rel in words for rel in rels)
    else:
        return any(rel in utterance for rel in rels)
    
def get_left_right_info(dataset_type, refer_dataset, line_number, use_original_viewdep_judge=True) -> bool:
    refer_data = refer_dataset[int(line_number)]
    utterance = refer_data['utterance']
    rels = [
        'left', 'right',
        'leftmost', 'rightmost'
    ]
    if use_original_viewdep_judge:
        words = set(utterance.split())
        return any(rel in words for rel in rels)
    else:
        return any(rel in utterance for rel in rels)
    
def get_ordinal_info(dataset_type, refer_dataset, line_number) -> bool:
    refer_data = refer_dataset[int(line_number)]
    utterance = refer_data['utterance']
    rels = [
        'from left', 'from right',
        'from the left', 'from the right'
    ]
    # words = set(utterance.split())
    return any(rel in utterance for rel in rels)

def get_correct_guess_info(dataset_type, refer_dataset, line_number) -> bool:
    # print(self.nr3d_data.keys())
    refer_data = refer_dataset[int(line_number)]
    if refer_data['correct_guess'] in ['True', 'TRUE', 'true']:
        return True
    else:
        return False
    
def analyse_result_sr3d(sr3d_dataset, result_path, use_original_viewdep_judge=True):
    # 本函数用于分析nr3d的结果
    # 首先处理path，如果是列表，就把所有的npy文件对应的np array合并
    if isinstance(result_path, list):
        for idx, path in enumerate(result_path):
            result_single = np.load(path, allow_pickle=True)
            if not idx:
                result = result_single
            else:
                result = np.vstack([result, result_single])
    else:
        result = np.load(result_path, allow_pickle=True)
    print("Sr3d results for:", result_path)
    # result=result[0:110,:]
    # 定义记录结果的字典
    accuracy_count = {
        "count_total": 0, "correct_count_total": 0,
        "count_easy": 0, "correct_count_easy": 0,
        "count_hard": 0, "correct_count_hard": 0,
        "count_view_dep": 0, "correct_count_view_dep": 0,
        "count_view_indep": 0, "correct_count_view_indep": 0,
        "count_left_right": 0, "correct_count_left_right": 0,
        "count_horizontal": 0, "correct_count_horizontal": 0,
        "count_vertical": 0, "correct_count_vertical": 0,
        "count_support": 0, "correct_count_support": 0,
        "count_between": 0, "correct_count_between": 0,
        "count_allocentric": 0, "correct_count_allocentric": 0
    }
    # 遍历，统计结果
    wrong_line_numbers = []
    for result_line in result:
        # 首先读取line_number
        line_number = result_line[0]  # 注意这里读进来是str
        # 如果是空行则跳过
        if result_line[0] == '':
            continue
        # 总数记数
        accuracy_count["count_total"] += 1
        # 获取easy信息并给easy的总数记数
        is_easy = get_easy_info('sr3d', sr3d_dataset, line_number)
        easy_setting = 'easy' if is_easy else 'hard'
        accuracy_count['count_%s' % easy_setting] += 1
        # 获取view_dependent信息并记数
        is_view_dep = get_view_dep_info('sr3d', sr3d_dataset, line_number, use_original_viewdep_judge)
        view_dep_setting = 'view_dep' if is_view_dep else 'view_indep'
        accuracy_count['count_%s' % view_dep_setting] += 1
        # 获取left_right信息并记数
        has_left_right = get_left_right_info('sr3d', sr3d_dataset, line_number, use_original_viewdep_judge)
        accuracy_count['count_left_right'] += 1 if has_left_right else 0
        # 五类空间关系记数
        reference_type = result_line[2]
        accuracy_count["count_" + reference_type] += 1
        # 给正确案例记数
        if result_line[5] == "True":
            accuracy_count["correct_count_total"] += 1
            accuracy_count['correct_count_%s' % easy_setting] += 1
            accuracy_count['correct_count_%s' % view_dep_setting] += 1
            accuracy_count['correct_count_left_right'] += 1 if has_left_right else 0
            accuracy_count["correct_count_" + reference_type] += 1
        else:
            wrong_line_numbers.append(eval(result_line[0]))
    # 打印准确率
    for name in ['total', 'easy', 'hard', 'view_dep', 'view_indep', 'left_right', 'horizontal', 'vertical', 'support', 'between', 'allocentric']:
        print(name + " accuracy:")
        correct = accuracy_count["correct_count_" + name]
        total = accuracy_count["count_" + name]
        percentage = -1 if total == 0 else correct / total * 100
        print("%.2f%% (%d/%d)" % (percentage, correct, total))
    print(f' & {round(accuracy_count["correct_count_horizontal"]/accuracy_count["count_horizontal"]*100, 1)} & {round(accuracy_count["correct_count_vertical"]/accuracy_count["count_vertical"]*100, 1)} & {round(accuracy_count["correct_count_support"]/accuracy_count["count_support"]*100, 1)} & {round(accuracy_count["correct_count_between"]/accuracy_count["count_between"]*100, 1)} & {round(accuracy_count["correct_count_allocentric"]/accuracy_count["count_allocentric"]*100, 1)} & {round(accuracy_count["correct_count_total"]/accuracy_count["count_total"]*100, 1)}\\\\')

def analyse_result_nr3d(nr3d_dataset, result_path, skip_human_wrong_cases=True, use_original_viewdep_judge=True):
    # 本函数用于分析nr3d的结果
    # 首先处理path，如果是列表，就把所有的npy文件对应的np array合并
    if isinstance(result_path, list):
        for idx, path in enumerate(result_path):
            result_single = np.load(path, allow_pickle=True)
            if not idx:
                result = result_single
            else:
                result = np.vstack([result, result_single])
    else:
        result = np.load(result_path, allow_pickle=True)
    print("Nr3d results for:", result_path)
    # result=result[0:110,:]
    # 定义记录结果的字典
    accuracy_count = {
        "count_total": 0, "correct_count_total": 0,
        "count_easy": 0, "correct_count_easy": 0,
        "count_hard": 0, "correct_count_hard": 0,
        "count_view_dep": 0, "correct_count_view_dep": 0,
        "count_view_indep": 0, "correct_count_view_indep": 0,
        "count_left_right": 0, "correct_count_left_right": 0,
        "count_ordinal": 0, "correct_count_ordinal": 0,  # from the left/right
        "count_use_object": 0, "correct_count_use_object": 0,
        "count_use_spatial": 0, "correct_count_use_spatial": 0,
        "count_use_color": 0, "correct_count_use_color": 0,
        "count_use_shape": 0, "correct_count_use_shape": 0,
    }
    # 遍历，统计结果
    wrong_line_numbers = []
    for result_line in result:
        # 首先读取line_number
        line_number = result_line[0]  # 注意这里读进来是str
        # 如果是空行则跳过
        if result_line[0] == '':
            continue
        # 按照nr3d的说明，如果是人类也没有答对（correct_guess==False)，则跳过
        if (not get_correct_guess_info('nr3d', nr3d_dataset, line_number)) and skip_human_wrong_cases:
            continue
        # 总数记数
        accuracy_count["count_total"] += 1
        # 获取easy信息并给easy的总数记数
        is_easy = get_easy_info('nr3d', nr3d_dataset, line_number)
        easy_setting = 'easy' if is_easy else 'hard'
        accuracy_count['count_%s' % easy_setting] += 1
        # 获取view_dependent信息并记数
        is_view_dep = get_view_dep_info('nr3d', nr3d_dataset, line_number, use_original_viewdep_judge)
        view_dep_setting = 'view_dep' if is_view_dep else 'view_indep'
        accuracy_count['count_%s' % view_dep_setting] += 1
        # 获取left_right信息并记数
        has_left_right = get_left_right_info('nr3d', nr3d_dataset, line_number, use_original_viewdep_judge)
        accuracy_count['count_left_right'] += 1 if has_left_right else 0
        # 获取left_right信息并记数
        is_ordinal = get_ordinal_info('nr3d', nr3d_dataset, line_number)
        accuracy_count['count_ordinal'] += 1 if is_ordinal else 0
        # use object,spatial,color,shape的记数
        use_lang_settings = ['use_object', 'use_spatial', 'use_color', 'use_shape']
        use_lang_settings_used = []
        for i in range(4):
            setting = use_lang_settings[i]
            if result_line[i + 7] in ['True', 'TRUE', 'true']:
                accuracy_count['count_%s' % setting] += 1
                use_lang_settings_used.append(setting)
        # 给正确案例记数
        if result_line[5] == "True":
            accuracy_count["correct_count_total"] += 1
            accuracy_count['correct_count_%s' % easy_setting] += 1
            accuracy_count['correct_count_%s' % view_dep_setting] += 1
            accuracy_count['correct_count_left_right'] += 1 if has_left_right else 0
            accuracy_count['correct_count_ordinal'] += 1 if is_ordinal else 0
            for setting in use_lang_settings_used:
                accuracy_count['correct_count_%s' % setting] += 1
        else:
            wrong_line_numbers.append(eval(result_line[0]))
    # 打印准确率
    for name in ['total', 'easy', 'hard', 'view_dep', 'view_indep', 'left_right', 'ordinal'] + use_lang_settings:
        print(name + " accuracy:")
        correct = accuracy_count["correct_count_" + name]
        total = accuracy_count["count_" + name]
        percentage = -1 if total == 0 else correct / total * 100
        print("%.2f%% (%d/%d)" % (percentage, correct, total))
        
def get_raw_label_2_nyu40_idx(self):
    type2class = {'cabinet': 0, 'bed': 1, 'chair': 2, 'sofa': 3, 'table': 4, 'door': 5,
                  'window': 6, 'bookshelf': 7, 'picture': 8, 'counter': 9, 'desk': 10, 'curtain': 11,
                  'refrigerator': 12, 'shower curtain': 13, 'toilet': 14, 'sink': 15, 'bathtub': 16, 'others': 17}
    scannet_labels = type2class.keys()
    scannet2label = {label: i for i, label in enumerate(scannet_labels)}  # 从上述18个label映射到idx的字典
    scannet_label_path = os.path.join(self.script_root, 'data', 'scannetv2-labels.combined.tsv')
    lines = [line.rstrip() for line in open(scannet_label_path)]
    lines = lines[1:]
    raw2label = {}
    for i in range(len(lines)):
        label_classes_set = set(scannet_labels)
        elements = lines[i].split('\t')
        raw_name = elements[1]
        nyu40_name = elements[7]
        if nyu40_name not in label_classes_set:
            raw2label[raw_name] = scannet2label['others']
        else:
            raw2label[raw_name] = scannet2label[nyu40_name]  # 从scannet中的raw_name映射到上述18个idx之一的字典
    return raw2label

def get_unique_info(data_root, scan_id, target_class) -> bool:
    # 本函数用于在结果npy文件没有记录scanrefer是否unique的情况下，找到这个信息
    # 读入事先准备好的物体信息，即npy文件
    # 做法参考了scanrefer的代码
    npy_path_train = os.path.join(data_root, "scannet_object_info", "objects_info", "objects_info_" + scan_id + ".npy")
    # npy_path_test=self.scannet_data_root+"/test/objects_info/objects_info_"+scan_id+".npy"
    if os.path.exists(npy_path_train):
        npy_path = npy_path_train
    # elif os.path.exists(npy_path_test):
    #     npy_path=npy_path_test
    else:
        print("object_info.npy file does not exist!!! scan_id:", scan_id)
        return None
    objects_info = np.load(npy_path, allow_pickle=True)  # objects_info是gt或3d segmentation得到的场景中所有物体的信息
    obj_idx_in_scene = []
    raw_label_2_nyu40_idx = get_raw_label_2_nyu40_idx(data_root)
    for obj in objects_info:
        raw_label = obj['label']
        idx = raw_label_2_nyu40_idx[raw_label]
        obj_idx_in_scene.append(idx)
    target_class = " ".join(target_class.split("_"))
    target_idx = raw_label_2_nyu40_idx[target_class]  # 将target class映射到18个idx
    is_unique = True if obj_idx_in_scene.count(target_idx) <= 1 else False
    return is_unique

def get_raw_label_2_nyu40_idx(data_root):
    type2class = {'cabinet': 0, 'bed': 1, 'chair': 2, 'sofa': 3, 'table': 4, 'door': 5,
                  'window': 6, 'bookshelf': 7, 'picture': 8, 'counter': 9, 'desk': 10, 'curtain': 11,
                  'refrigerator': 12, 'shower curtain': 13, 'toilet': 14, 'sink': 15, 'bathtub': 16, 'others': 17}
    scannet_labels = type2class.keys()
    scannet2label = {label: i for i, label in enumerate(scannet_labels)}  # 从上述18个label映射到idx的字典
    scannet_label_path = os.path.join(data_root, 'scannetv2-labels.combined.tsv')
    lines = [line.rstrip() for line in open(scannet_label_path)]
    lines = lines[1:]
    raw2label = {}
    for i in range(len(lines)):
        label_classes_set = set(scannet_labels)
        elements = lines[i].split('\t')
        raw_name = elements[1]
        nyu40_name = elements[7]
        if nyu40_name not in label_classes_set:
            raw2label[raw_name] = scannet2label['others']
        else:
            raw2label[raw_name] = scannet2label[nyu40_name]  # 从scannet中的raw_name映射到上述18个idx之一的字典
    return raw2label

def analyse_result_scanrefer(data_root, result_path, report_none_gt_error=True, use_gt_box=True, iou_thr=0.5):
    # 本函数用于分析scanrefer的结果
    # 首先处理path，如果是列表，就把所有的npy文件对应的np array合并
    if isinstance(result_path, list):
        for idx, path in enumerate(result_path):
            result_single = np.load(path, allow_pickle=True)
            if not idx:
                result = result_single
            else:
                result = np.vstack([result, result_single])
    else:
        result = np.load(result_path, allow_pickle=True)
    print("Scanrefer results for:", result_path)
    # 定义记录结果的字典
    accuracy_count = {
        "count_total": 0, "correct_count_total_25": 0, "correct_count_total_50": 0,
        "count_unique": 0, "correct_count_unique_25": 0, "correct_count_unique_50": 0,
        "count_multiple": 0, "correct_count_multiple_25": 0, "correct_count_multiple_50": 0,
    }
    # 遍历结果，在accuracy_count中记录相应数据
    iou_list = []
    correct_answer_exist_count = 0
    wrong_line_numbers = []
    wrong_line_numbers_except = []
    for result_line in result:
        # 如果是空行则跳过
        if result_line[0] == '':
            continue
        # 读入scan_id和target_class，并获取是否unique
        scan_id = result_line[1]
        target_class = result_line[8]
        if target_class == 'toilet_paper_dispense':
            target_class = 'toilet_paper_dispenser'
        is_unique = get_unique_info(data_root, scan_id, target_class)
        # 读入iou
        iou = eval(result_line[7])
        # 总数记数
        accuracy_count["count_total"] += 1
        if is_unique:
            accuracy_count["count_unique"] += 1
        else:
            accuracy_count["count_multiple"] += 1
        # iou超过0.25/0.5则给正确数记数
        if iou >= 0.5:
            accuracy_count["correct_count_total_50"] += 1
            if is_unique:
                accuracy_count["correct_count_unique_50"] += 1
            else:
                accuracy_count["correct_count_multiple_50"] += 1
        if iou >= 0.25:
            accuracy_count["correct_count_total_25"] += 1
            if is_unique:
                accuracy_count["correct_count_unique_25"] += 1
            else:
                accuracy_count["correct_count_multiple_25"] += 1
        else:
            wrong_line_numbers.append(eval(result_line[0]))
        iou_list.append(iou)
        # 这里还需要记录该案例是否有可能找到正确答案，改为用max_iou比较
        if eval(result_line[10]) >= iou_thr:
            correct_answer_exist_count += 1
            if iou <= iou_thr:
                wrong_line_numbers_except.append(eval(result_line[0]))  # 记录有正确答案的情况下的错误案例
    # print("wrong cases line_numbers:",wrong_line_numbers)
    # print("wrong cases line_numbers:",wrong_line_numbers_except)
    # 不同setting和k的Acc@k
    for setting in ['total', 'multiple', 'unique']:
        for thr in [50, 25]:
            correct = accuracy_count["correct_count_%s_%d" % (setting, thr)]
            total = accuracy_count["count_%s" % setting]
            percentage = -1 if total == 0 else correct / total * 100
            print("Acc@%.2f %s:" % (thr / 100, setting))
            print("%.2f%% (%d/%d)" % (percentage, correct, total))
    # 平均iou
    print("average iou:")
    print("%.3f" % np.average(iou_list))
    # groupfree没提供正确答案的比例
    if report_none_gt_error and not use_gt_box:
        total = accuracy_count["count_total"]
        correct = accuracy_count["correct_count_total_50"]
        wrong = total - correct
        no_correct_answer = total - correct_answer_exist_count
        percentage = "-" if wrong == 0 else no_correct_answer / wrong * 100
        print("Percentage of error caused by 'no correct answer provided by Group Free':")
        print("%.2f%% (%d/%d)" % (percentage, no_correct_answer, wrong))
        # 去除上述情况后的Acc@k
        percentage = "-" if correct_answer_exist_count == 0 else correct / correct_answer_exist_count * 100
        print("Acc@%.2f without such cases:" % iou_thr)
        print("%.2f%% (%d/%d)" % (percentage, correct, correct_answer_exist_count))
        
def analyse_result(dataset_type, refer_dataset_path, result_path, config):
    data_root, use_original_viewdep_judge, scanrefer_use_gt_box, scanrefer_iou_thr = config
    refer_dataset = load_refer_dataset_pure(dataset_type, refer_dataset_path)
    if dataset_type == 'sr3d':
        analyse_result_sr3d(refer_dataset, result_path, use_original_viewdep_judge=use_original_viewdep_judge)
    elif dataset_type == 'nr3d':
        analyse_result_nr3d(refer_dataset, result_path, skip_human_wrong_cases=True, use_original_viewdep_judge=use_original_viewdep_judge)
    else:
        analyse_result_scanrefer(data_root, result_path, report_none_gt_error=True, use_gt_box=scanrefer_use_gt_box, iou_thr=scanrefer_iou_thr)
    return