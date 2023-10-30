
confs_nr3d={
    # nr3d
    0:{
        'dataset':'nr3d',
        'model':'gpt-4',  # 4 no principle
        'result_folder_name':'eval_results_nr3d_4_np_testset/',
        'use_principle':False,
        'refer_dataset_path':"./data/referit3d/nr3d_test_sampled1000.csv",
        'use_gt_box':True,
        'use_priority':False,
        'use_code_interpreter':True
    },
    1:{
        'dataset':'nr3d',
        'model':'gpt-4',  # 4 with principle
        'result_folder_name':'eval_results_nr3d_4_p_testset/',
        'use_principle':True,
        'refer_dataset_path':"./data/referit3d/nr3d_test_sampled1000.csv",
        'use_gt_box':True,
        'use_priority':False,
        'use_code_interpreter':True
    },
    2:{
        'dataset':'nr3d',
        'model': 'ft:gpt-3.5-turbo-0613:ripl:ref-suc573-cor233:7z8ztnAu', # 3.5 ft suc573加cor233
        'result_folder_name':'eval_results_nr3d_35ft_np_sc_testset/',
        'use_principle':False,
        'refer_dataset_path':"./data/referit3d/nr3d_test_sampled1000.csv",
        'use_gt_box':True,
        'use_priority':False,
        'use_code_interpreter':True
    },
    3:{
        'dataset':'nr3d',
        'model':'ft:gpt-3.5-turbo-0613:ripl:refer-succ-166-nr:7yp0M6Nn', #3.5 ft success573
        'result_folder_name':'eval_results_nr3d_35ft_np_c_testset/',
        'use_principle':False,
        'refer_dataset_path':"./data/referit3d/nr3d_test_sampled1000.csv",
        'use_gt_box':True,
        'use_priority':False,
        'use_code_interpreter':True
    },
    4:{
        'dataset':'nr3d',
        'model':'gpt-3.5-turbo', # 3.5 zero shot no principle
        'result_folder_name':'eval_results_nr3d_35_np_testset/',
        'use_principle':False,
        'refer_dataset_path':"./data/referit3d/nr3d_test_sampled1000.csv",
        'use_gt_box':True,
        'use_priority':False,
        'use_code_interpreter':True
    },
    5:{
        'dataset':'nr3d',
        'model':'gpt-3.5-turbo', # 3.5 zero shot with principle
        'result_folder_name':'eval_results_nr3d_35_p_testset/',
        'use_principle':True,
        'refer_dataset_path':"./data/referit3d/nr3d_test_sampled1000.csv",
        'use_gt_box':True,
        'use_priority':False,
        'use_code_interpreter':True
    },
    6:{
        'dataset':'nr3d',
        'model':'gpt-4', # nr3d gpt-4 train set
        'result_folder_name':'eval_results_nr3d_4_p_trainset/',
        'use_principle':True,
        'refer_dataset_path':"./data/referit3d/nr3d_train_sampled1000.csv",
        'use_gt_box':True,
        'use_priority':False,
        'use_code_interpreter':True
    },
}

confs_sr3d={
    # sr3d
    0:{
        'dataset':'sr3d',
        'model':'gpt-4', # 4 no principle
        'result_folder_name':'eval_results_sr3d_4_np_testset/',
        'use_principle':False,
        'refer_dataset_path':"./data/referit3d/sr3d_test_sampled1000.csv",
        'use_gt_box':True,
        'use_priority':False,
        'use_code_interpreter':True
    },
    1:{
        'dataset':'sr3d',
        'model':'gpt-4', # 4 with principle
        'result_folder_name':'eval_results_sr3d_4_p_testset/',
        'use_principle':True,
        'refer_dataset_path':"./data/referit3d/sr3d_test_sampled1000.csv",
        'use_gt_box':True,
        'use_priority':False,
        'use_code_interpreter':True
    },
    2:{
        'dataset':'sr3d',
        'model':'gpt-3.5-turbo', # 3.5 zero shot no principle
        'result_folder_name':'eval_results_sr3d_35_np_testset/',
        'use_principle':False,
        'refer_dataset_path':"./data/referit3d/sr3d_test_sampled1000.csv",
        'use_gt_box':True,
        'use_priority':False,
        'use_code_interpreter':True
    },
    3:{
        'dataset':'sr3d',
        'model':'gpt-3.5-turbo', # 3.5 zero shot with principle
        'result_folder_name':'eval_results_sr3d_35_p_testset/',
        'use_principle':True,
        'refer_dataset_path':"./data/referit3d/sr3d_test_sampled1000.csv",
        'use_gt_box':True,
        'use_priority':False,
        'use_code_interpreter':True
    },
    4:{
        'dataset':'sr3d',
        'model': 'ft:gpt-3.5-turbo-0613:ripl:ref-suc573-cor233:7z8ztnAu', # 3.5 ft suc573加cor233
        'result_folder_name':'eval_results_sr3d_35_np_sc_testset/',
        'use_principle':False,
        'refer_dataset_path':"./data/referit3d/sr3d_test_sampled1000.csv",
        'use_gt_box':True,
        'use_priority':False,
        'use_code_interpreter':True
    },
    5:{
        'dataset':'sr3d',
        'model':'ft:gpt-3.5-turbo-0613:ripl:refer-succ-166-nr:7yp0M6Nn', #3.5 ft success573
        'result_folder_name':'eval_results_sr3d_35_np_c_testset/',
        'use_principle':False,
        'refer_dataset_path':"./data/referit3d/sr3d_test_sampled1000.csv",
        'use_gt_box':True,
        'use_priority':False,
        'use_code_interpreter':True
    },
    #sr3d
    6:{
        'dataset':'sr3d',
        'model':'gpt-4',  # sr3d no code interpreter, no principle
        'result_folder_name':'eval_results_sr3d_4_np_as_nocode_testset/',
        'use_principle':False,
        'refer_dataset_path':"./data/referit3d/sr3d_test_assembled30x5.csv",
        'use_gt_box':True,
        'use_priority':False,
        'use_code_interpreter':False
    },
    7:{
        'dataset':'sr3d',
        'model':'gpt-4',  # sr3d with code interpreter, no principle
        'result_folder_name':'eval_results_sr3d_4_np_as_code_testset/',
        'use_principle':False,
        'refer_dataset_path':"./data/referit3d/sr3d_test_assembled30x5.csv",
        'use_gt_box':True,
        'use_priority':False,
        'use_code_interpreter':True
    },
    8:{
        'dataset':'sr3d',
        'model':'gpt-4',  # sr3d no code interpreter, with principle
        'result_folder_name':'eval_results_sr3d_4_p_as_nocode_testset/',
        'use_principle':True,
        'refer_dataset_path':"./data/referit3d/sr3d_test_assembled30x5.csv",
        'use_gt_box':True,
        'use_priority':False,
        'use_code_interpreter':False
    },
    9:{
        'dataset':'sr3d',
        'model':'gpt-4',  # sr3d with code interpreter, with principle
        'result_folder_name':'eval_results_sr3d_4_p_as_code_testset/',
        'use_principle':True,
        'refer_dataset_path':"./data/referit3d/sr3d_test_assembled30x5.csv",
        'use_gt_box':True,
        'use_priority':False,
        'use_code_interpreter':True
    },
}

confs_scanrefer={
    # scanrefer
    0:{
        'dataset':'scanrefer',
        'model':'gpt-4', # 4 with gt box on val set
        'result_folder_name':'eval_results_scanrefer_4_p_gtbox_valset/',
        'use_principle':True,
        'refer_dataset_path':"./data/scanrefer/ScanRefer_filtered_val_sampled1000.json",
        'use_gt_box':True,
        'use_priority':False,
        'use_code_interpreter':True
    },
    1:{
        'dataset':'scanrefer',
        'model':'gpt-4', # 4 with gt box on train set
        'result_folder_name':'eval_results_scanrefer_2stages/',
        'use_principle':True,
        'refer_dataset_path':"./data/scanrefer/ScanRefer_filtered_train_sampled1000.json",
        'use_gt_box':True,
        'use_priority':False,
        'use_code_interpreter':True
    },
    2:{
        'dataset':'scanrefer',
        'model':'gpt-4', # 4 with mask3d 200c on val set
        'result_folder_name':'eval_results_scanrefer_4_p_mask3d_200c_valset/',
        'use_principle':True,
        'refer_dataset_path':"./data/scanrefer/ScanRefer_filtered_val_sampled1000.json",
        'use_gt_box':False,
        'use_priority':False,
        'use_code_interpreter':True,
        'tool':'mask3d_200c'
    },
    3:{
        'dataset':'scanrefer',
        'model':'gpt-4', # 4 with mask3d 20c on val set
        'result_folder_name':'eval_results_scanrefer_4_p_mask3d_20c_valset/',
        'use_principle':True,
        'refer_dataset_path':"./data/scanrefer/ScanRefer_filtered_val_sampled1000.json",
        'use_gt_box':False,
        'use_priority':False,
        'use_code_interpreter':True,
        'tool':'mask3d_20c'
    },
    4:{
        'dataset':'scanrefer',
        'model':'gpt-4', # 4 with gf val set
        'result_folder_name':'eval_results_scanrefer_4_p_gf_valset/',
        'use_principle':True,
        'refer_dataset_path':"./data/scanrefer/ScanRefer_filtered_val_sampled1000.json", # 改正
        'use_gt_box':False,
        'use_priority':False,
        'use_code_interpreter':True,
        'tool':'gf'
    },
    5:{
        'dataset':'scanrefer',
        'model':'gpt-4', # 4 with gf all sets sample 50
        'result_folder_name':'eval_results_scanrefer_2stages/',
        'use_principle':True,
        'refer_dataset_path':"./data/scanrefer/ScanRefer_filtered_sampled50.json",
        'use_gt_box':False,
        'use_priority':False,
        'use_code_interpreter':True,
        'tool':'gf'
    },
    6:{
        'dataset':'scanrefer',
        'model':'gpt-4', # 4 with gf priority
        'result_folder_name':'eval_results_scanrefer_4_p_gf_prio_valset/',
        'use_principle':True,
        'refer_dataset_path':"./data/scanrefer/ScanRefer_filtered_val_sampled1000.json",
        'use_gt_box':False,
        'use_priority':True,
        'use_code_interpreter':True,
        'tool':'gf'
    },
}