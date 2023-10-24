
test_modes_nr3d={
    # nr3d
    0:{
        'dataset':'nr3d',
        'model':'gpt-4',  # 4 no principle
        'result_folder_name':'eval_results_nr3d_4_np_testset/',
        'use_principle':False,
        'formatted_time_list':['2023-09-15-20-11-51','2023-09-15-20-59-50','2023-09-15-21-26-37','2023-09-15-23-17-24'],
        'refer_dataset_path':"/share/data/ripl/vincenttann/sr3d/data/referit3d/nr3d_test_sampled1000.csv",
        'use_gt_box':True,
        'use_priority':False,
        'use_code_interpreter':True
    },
    1:{
        'dataset':'nr3d',
        'model':'gpt-4',  # 4 with principle
        'result_folder_name':'eval_results_nr3d_4_p_testset/',
        'use_principle':True,
        'formatted_time_list':['2023-09-15-18-17-39','2023-09-15-19-02-07','2023-09-15-19-47-00'],
        'refer_dataset_path':"/share/data/ripl/vincenttann/sr3d/data/referit3d/nr3d_test_sampled1000.csv",
        'use_gt_box':True,
        'use_priority':False,
        'use_code_interpreter':True
    },
    2:{
        'dataset':'nr3d',
        'model': 'ft:gpt-3.5-turbo-0613:ripl:ref-suc573-cor233:7z8ztnAu', # 3.5 ft suc573加cor233
        'result_folder_name':'eval_results_nr3d_35ft_np_sc_testset/',
        'use_principle':False,
        'formatted_time_list':['2023-09-15-15-14-28','2023-09-15-18-06-29','2023-09-15-17-13-51','2023-09-15-17-14-30'],
        'refer_dataset_path':"/share/data/ripl/vincenttann/sr3d/data/referit3d/nr3d_test_sampled1000.csv",
        'use_gt_box':True,
        'use_priority':False,
        'use_code_interpreter':True
    },
    3:{
        'dataset':'nr3d',
        'model':'ft:gpt-3.5-turbo-0613:ripl:refer-succ-166-nr:7yp0M6Nn', #3.5 ft success573
        'result_folder_name':'eval_results_nr3d_35ft_np_c_testset/',
        'use_principle':False,
        'formatted_time_list':['2023-09-15-19-02-56','2023-09-15-19-28-07','2023-09-15-19-34-53'],
        'refer_dataset_path':"/share/data/ripl/vincenttann/sr3d/data/referit3d/nr3d_test_sampled1000.csv",
        'use_gt_box':True,
        'use_priority':False,
        'use_code_interpreter':True
    },
    4:{
        'dataset':'nr3d',
        'model':'gpt-3.5-turbo', # 3.5 zero shot no principle
        'result_folder_name':'eval_results_nr3d_35_np_testset/',
        'use_principle':False,
        'formatted_time_list':['2023-09-15-18-22-33','2023-09-15-18-24-33','2023-09-15-18-25-34'],
        'refer_dataset_path':"/share/data/ripl/vincenttann/sr3d/data/referit3d/nr3d_test_sampled1000.csv",
        'use_gt_box':True,
        'use_priority':False,
        'use_code_interpreter':True
    },
    5:{
        'dataset':'nr3d',
        'model':'gpt-3.5-turbo', # 3.5 zero shot with principle
        'result_folder_name':'eval_results_nr3d_35_p_testset/',
        'use_principle':True,
        'formatted_time_list':['2023-09-15-20-26-19','2023-09-15-20-46-00','2023-09-15-20-58-52'],
        'refer_dataset_path':"/share/data/ripl/vincenttann/sr3d/data/referit3d/nr3d_test_sampled1000.csv",
        'use_gt_box':True,
        'use_priority':False,
        'use_code_interpreter':True
    },
}

test_modes_sr3d={
    # sr3d
    0:{
        'dataset':'sr3d',
        'model':'gpt-4', # 4 no principle
        'result_folder_name':'eval_results_sr3d_4_np_testset/',
        'use_principle':False,
        'formatted_time_list':['2023-09-16-15-52-57'],
        'refer_dataset_path':"/share/data/ripl/vincenttann/sr3d/data/referit3d/sr3d_test_sampled1000.csv",
        'use_gt_box':True,
        'use_priority':False,
        'use_code_interpreter':True
    },
    1:{
        'dataset':'sr3d',
        'model':'gpt-4', # 4 with principle
        'result_folder_name':'eval_results_sr3d_4_p_testset/',
        'use_principle':True,
        # 'formatted_time_list':['2023-09-15-22-58-19','2023-09-16-00-05-43','2023-09-16-14-28-01'], #wasted
        'formatted_time_list':['2023-09-16-00-37-29','2023-09-16-00-37-50','2023-09-16-00-58-50','2023-09-16-14-27-46'],
        'refer_dataset_path':"/share/data/ripl/vincenttann/sr3d/data/referit3d/sr3d_test_sampled1000.csv",
        'use_gt_box':True,
        'use_priority':False,
        'use_code_interpreter':True
    },
    2:{
        'dataset':'sr3d',
        'model':'gpt-3.5-turbo', # 3.5 zero shot no principle
        'result_folder_name':'eval_results_sr3d_35_np_testset/',
        'use_principle':False,
        'formatted_time_list':['2023-09-16-16-00-42'],
        'refer_dataset_path':"/share/data/ripl/vincenttann/sr3d/data/referit3d/sr3d_test_sampled1000.csv",
        'use_gt_box':True,
        'use_priority':False,
        'use_code_interpreter':True
    },
    3:{
        'dataset':'sr3d',
        'model':'gpt-3.5-turbo', # 3.5 zero shot with principle
        'result_folder_name':'eval_results_sr3d_35_p_testset/',
        'use_principle':True,
        'formatted_time_list':['2023-09-16-16-01-16'],
        'refer_dataset_path':"/share/data/ripl/vincenttann/sr3d/data/referit3d/sr3d_test_sampled1000.csv",
        'use_gt_box':True,
        'use_priority':False,
        'use_code_interpreter':True
    },
    4:{
        'dataset':'sr3d',
        'model': 'ft:gpt-3.5-turbo-0613:ripl:ref-suc573-cor233:7z8ztnAu', # 3.5 ft suc573加cor233
        'result_folder_name':'eval_results_sr3d_35_np_sc_testset/',
        'use_principle':False,
        'formatted_time_list':['2023-09-16-17-03-26'],
        'refer_dataset_path':"/share/data/ripl/vincenttann/sr3d/data/referit3d/sr3d_test_sampled1000.csv",
        'use_gt_box':True,
        'use_priority':False,
        'use_code_interpreter':True
    },
    5:{
        'dataset':'sr3d',
        'model':'ft:gpt-3.5-turbo-0613:ripl:refer-succ-166-nr:7yp0M6Nn', #3.5 ft success573
        'result_folder_name':'eval_results_sr3d_35_np_c_testset/',
        'use_principle':False,
        'formatted_time_list':['2023-09-16-17-03-49'],
        'refer_dataset_path':"/share/data/ripl/vincenttann/sr3d/data/referit3d/sr3d_test_sampled1000.csv",
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
        'formatted_time_list':['2023-09-16-19-06-14'],
        'refer_dataset_path':"/share/data/ripl/vincenttann/sr3d/data/referit3d/sr3d_test_assembled30x5.csv",
        'use_gt_box':True,
        'use_priority':False,
        'use_code_interpreter':False
    },
    7:{
        'dataset':'sr3d',
        'model':'gpt-4',  # sr3d with code interpreter, no principle
        'result_folder_name':'eval_results_sr3d_4_np_as_code_testset/',
        'use_principle':False,
        'formatted_time_list':['2023-09-16-19-09-37'],
        'refer_dataset_path':"/share/data/ripl/vincenttann/sr3d/data/referit3d/sr3d_test_assembled30x5.csv",
        'use_gt_box':True,
        'use_priority':False,
        'use_code_interpreter':True
    },
    8:{
        'dataset':'sr3d',
        'model':'gpt-4',  # sr3d no code interpreter, with principle
        'result_folder_name':'eval_results_sr3d_4_p_as_nocode_testset/',
        'use_principle':True,
        'formatted_time_list':['2023-09-16-20-12-59'],
        'refer_dataset_path':"/share/data/ripl/vincenttann/sr3d/data/referit3d/sr3d_test_assembled30x5.csv",
        'use_gt_box':True,
        'use_priority':False,
        'use_code_interpreter':False
    },
    9:{
        'dataset':'sr3d',
        'model':'gpt-4',  # sr3d with code interpreter, with principle
        'result_folder_name':'eval_results_sr3d_4_p_as_code_testset/',
        'use_principle':True,
        # 'formatted_time_list':['2023-09-16-19-10-01','2023-09-16-21-31-15','2023-09-16-22-07-35'], #完整的，但是support不好
        # 'formatted_time_list':['2023-09-16-22-45-55'],
        # 'formatted_time_list':['2023-09-16-23-14-27','2023-09-16-23-19-44'], # support
        'formatted_time_list':['2023-09-16-23-18-01'], # vertical

        'refer_dataset_path':"/share/data/ripl/vincenttann/sr3d/data/referit3d/sr3d_test_assembled30x5.csv",
        'use_gt_box':True,
        'use_priority':False,
        'use_code_interpreter':True
    },
}

test_modes_scanrefer={
    # scanrefer
    0:{
        'dataset':'scanrefer',
        'model':'gpt-4', # 4 with gt box on val set
        'result_folder_name':'eval_results_scanrefer_4_p_gtbox_valset/',
        'use_principle':True,
        # 'formatted_time_list':['2023-09-16-19-18-50','2023-09-16-20-22-10'],
        'formatted_time_list':['2023-09-16-22-05-20','2023-09-16-22-05-32','2023-09-16-22-05-59'], # 有79个
        'refer_dataset_path':"/share/data/ripl/vincenttann/sr3d/data/scanrefer/ScanRefer_filtered_val_sampled1000.json",
        'use_gt_box':True,
        'use_priority':False,
        'use_code_interpreter':True
    },
    1:{
        'dataset':'scanrefer',
        'model':'gpt-4', # 4 with gt box on train set
        'result_folder_name':'eval_results_scanrefer_2stages/',
        'use_principle':True,
        # 'formatted_time_list':["2023-09-13-01-05-25","2023-09-13-01-05-36","2023-09-13-01-05-49","2023-09-13-01-05-59","2023-09-13-17-18-48","2023-09-13-17-19-05","2023-09-13-17-19-19","2023-09-13-17-19-33"] ,
        'formatted_time_list':["2023-09-13-01-05-25","2023-09-13-01-05-36","2023-09-13-01-05-49","2023-09-13-01-05-59","2023-09-13-14-38-48","2023-09-13-14-38-55","2023-09-13-14-38-59","2023-09-13-14-39-03"] ,
        'refer_dataset_path':"/share/data/ripl/vincenttann/sr3d/data/scanrefer/ScanRefer_filtered_train_sampled1000.json",
        'use_gt_box':True,
        'use_priority':False,
        'use_code_interpreter':True
    },
    2:{
        'dataset':'scanrefer',
        'model':'gpt-4', # 4 with mask3d 200c on val set
        'result_folder_name':'eval_results_scanrefer_4_p_mask3d_200c_valset/',
        'use_principle':True,
        # 'formatted_time_list':None, #['2023-09-16-01-25-55','2023-09-16-01-26-14'],
        # 'formatted_time_list':['2023-10-05-14-30-55','2023-10-05-14-55-37'], # 10月5号测的
        'formatted_time_list':['2023-10-06-15-28-03'], # 10月6号测，修正了代码，并加了non-max suppression
        'refer_dataset_path':"/share/data/ripl/vincenttann/sr3d/data/scanrefer/ScanRefer_filtered_val_sampled1000.json",
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
        'formatted_time_list':['2023-10-05-15-57-37'],
        'refer_dataset_path':"/share/data/ripl/vincenttann/sr3d/data/scanrefer/ScanRefer_filtered_val_sampled1000.json",
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
        # 'formatted_time_list':['2023-09-15-23-51-36','2023-09-16-00-02-17','2023-09-16-00-30-09'],  #这些跑的好像也是scanrefer的train
        'formatted_time_list':['2023-10-05-15-22-55',],
        # 'refer_dataset_path':"/share/data/ripl/vincenttann/sr3d/data/scanrefer/ScanRefer_filtered_train_sampled1000.json", #这里好像把val写成train了...
        'refer_dataset_path':"/share/data/ripl/vincenttann/sr3d/data/scanrefer/ScanRefer_filtered_val_sampled1000.json", # 改正
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
        # 'formatted_time_list':['2023-09-01-00-30-50'],
        # 'formatted_time_list':['2023-08-31-22-32-34'],
        'formatted_time_list':['2023-08-31-18-26-47'],
        'refer_dataset_path':"/share/data/ripl/vincenttann/sr3d/data/scanrefer/ScanRefer_filtered_sampled50.json",
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
        'formatted_time_list':['2023-09-16-02-43-49','2023-09-16-02-44-14','2023-09-16-02-44-37'],
        # 'formatted_time_list':['2023-09-16-02-44-37'],
        'refer_dataset_path':"/share/data/ripl/vincenttann/sr3d/data/scanrefer/ScanRefer_filtered_val_sampled1000.json",
        'use_gt_box':False,
        'use_priority':True,
        'use_code_interpreter':True,
        'tool':'gf'
    },


}