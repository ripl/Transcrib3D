
confs_nr3d = {
    # nr3d
    0: {
        'dataset_type': 'nr3d',
        'model': 'gpt-4',  # 4 no principle
        'result_folder_name': 'eval_results_nr3d_4_np_testset/',
        'use_principle': False,
        'refer_dataset_path': "./data/referit3d/nr3d_test_sampled1000.csv",
        'use_gt_box': True,
        'use_priority': False,
        'use_code_interpreter': True
    },
    1: {
        'dataset_type': 'nr3d',
        'model': 'gpt-4',  # 4 with principle
        'result_folder_name': 'eval_results_nr3d_4_p_testset/',
        'use_principle': True,
        'refer_dataset_path': "./data/referit3d/nr3d_test_sampled1000.csv",
        'use_gt_box': True,
        'use_priority': False,
        'use_code_interpreter': True
    },
    2: {
        'dataset_type': 'nr3d',
        'model': 'ft:gpt-3.5-turbo-0613:ripl:ref-suc573-cor233:7z8ztnAu',  # 3.5 ft suc573加cor233
        'result_folder_name': 'eval_results_nr3d_35ft_np_sc_testset/',
        'use_principle': False,
        'refer_dataset_path': "./data/referit3d/nr3d_test_sampled1000.csv",
        'use_gt_box': True,
        'use_priority': False,
        'use_code_interpreter': True
    },
    3: {
        'dataset_type': 'nr3d',
        'model': 'ft:gpt-3.5-turbo-0613:ripl:refer-succ-166-nr:7yp0M6Nn',  # 3.5 ft success573
        'result_folder_name': 'eval_results_nr3d_35ft_np_c_testset/',
        'use_principle': False,
        'refer_dataset_path': "./data/referit3d/nr3d_test_sampled1000.csv",
        'use_gt_box': True,
        'use_priority': False,
        'use_code_interpreter': True
    },
    4: {
        'dataset_type': 'nr3d',
        'model': 'gpt-3.5-turbo',  # 3.5 zero shot no principle
        'result_folder_name': 'eval_results_nr3d_35_np_testset/',
        'use_principle': False,
        'refer_dataset_path': "./data/referit3d/nr3d_test_sampled1000.csv",
        'use_gt_box': True,
        'use_priority': False,
        'use_code_interpreter': True
    },
    5: {
        'dataset_type': 'nr3d',
        'model': 'gpt-3.5-turbo',  # 3.5 zero shot with principle
        'result_folder_name': 'eval_results_nr3d_35_p_testset/',
        'use_principle': True,
        'refer_dataset_path': "./data/referit3d/nr3d_test_sampled1000.csv",
        'use_gt_box': True,
        'use_priority': False,
        'use_code_interpreter': True
    },
    6: {
        'dataset_type': 'nr3d',
        'model': 'gpt-4',  # nr3d gpt-4 train set
        'result_folder_name': 'eval_results_nr3d_4_p_trainset/',
        'use_principle': True,
        'refer_dataset_path': "./data/referit3d/nr3d_train_sampled1000.csv",
        'use_gt_box': True,
        'use_priority': False,
        'use_code_interpreter': True
    },
    106: {
        'dataset_type': 'nr3d',
        'model': 'gpt-4-1106-preview',  # nr3d gpt-4-turbo train set
        'result_folder_name': 'eval_results_nr3d_4_p_trainset/',
        'use_principle': True,
        'refer_dataset_path': "./data/referit3d/nr3d_train_sampled1000.csv",
        'use_gt_box': True,
        'use_priority': False,
        'use_code_interpreter': True
    },
    7: {
        'dataset_type': 'nr3d',
        'model': 'meta-llama/Llama-2-7b-chat-hf',
        'result_folder_name': 'eval_results_nr3d_llama2_7b_chat_p_testset/',
        'use_principle': True,
        'refer_dataset_path': "./data/referit3d/nr3d_test_sampled1000.csv",
        'use_gt_box': True,
        'use_priority': False,
        'use_code_interpreter': False
    },
    8: {
        'dataset_type': 'nr3d',
        'model': 'meta-llama/Llama-2-13b-chat-hf',
        'result_folder_name': 'eval_results_nr3d_llama2_13b_chat_p_testset/',
        'use_principle': True,
        'refer_dataset_path': "./data/referit3d/nr3d_test_sampled1000.csv",
        'use_gt_box': True,
        'use_priority': False,
        'use_code_interpreter': False
    },
    9: {
        'dataset_type': 'nr3d',
        'model': 'meta-llama/Llama-2-70b-chat-hf',
        'result_folder_name': 'eval_results_nr3d_llama2_70b_chat_p_testset/',
        'use_principle': True,
        'refer_dataset_path': "./data/referit3d/nr3d_test_sampled1000.csv",
        'use_gt_box': True,
        'use_priority': False,
        'use_code_interpreter': False
    },
    10: {
        'dataset_type': 'nr3d',
        'model': 'codellama/CodeLlama-7b-Instruct-hf',
        'result_folder_name': 'eval_results_nr3d_codellama_7b_instruct_p_testset/',
        'use_principle': True,
        'refer_dataset_path': "./data/referit3d/nr3d_test_sampled1000.csv",
        'use_gt_box': True,
        'use_priority': False,
        'use_code_interpreter': False
    },
    11: {
        'dataset_type': 'nr3d',
        'model': 'codellama/CodeLlama-13b-Instruct-hf',
        'result_folder_name': 'eval_results_nr3d_codellama_13b_instruct_p_testset/',
        'use_principle': True,
        'refer_dataset_path': "./data/referit3d/nr3d_test_sampled1000.csv",
        'use_gt_box': True,
        'use_priority': False,
        'use_code_interpreter': False
    },
    12: {
        'dataset_type': 'nr3d',
        'model': 'codellama/CodeLlama-34b-Instruct-hf',
        'result_folder_name': 'eval_results_nr3d_codellama_34b_instruct_p_testset/',
        'use_principle': True,
        'refer_dataset_path': "./data/referit3d/nr3d_test_sampled1000.csv",
        'use_gt_box': True,
        'use_priority': False,
        'use_code_interpreter': False
    },
    # 13:{
    #     'dataset_type':'nr3d',
    #     'model':'mistralai/Mistral-7B-Instruct-v0.1',
    #     'result_folder_name':'eval_results_nr3d_mistralai_7b_instruct_p_testset/',
    #     'use_principle':True,
    #     'refer_dataset_path':"./data/referit3d/nr3d_test_sampled1000.csv",
    #     'use_gt_box':True,
    #     'use_priority':False,
    #     'use_code_interpreter':False
    # },
    14: {
        'dataset_type': 'nr3d',
        'model': 'gpt-4',  # 4 with principle, no code interpreter
        'result_folder_name': 'eval_results_nr3d_4_p_testset/',
        'use_principle': True,
        'refer_dataset_path': "./data/referit3d/nr3d_test_sampled1000.csv",
        'use_gt_box': True,
        'use_priority': False,
        'use_code_interpreter': False
    },
    15: {
        'dataset_type': 'nr3d',
        'model': 'gpt-3.5-turbo',  # 3.5 with principle, no code interpreter
        'result_folder_name': 'eval_results_nr3d_35_p_testset/',
        'use_principle': True,
        'refer_dataset_path': "./data/referit3d/nr3d_test_sampled1000.csv",
        'use_gt_box': True,
        'use_priority': False,
        'use_code_interpreter': False
    },
    16: {
        'dataset_type': 'nr3d',
        'model': 'gpt-4',  # 4 with principle, no code interpreter, no object filter
        'result_folder_name': 'eval_results_nr3d_4_p_no_code_no_filter_testset/',
        'use_principle': True,
        'refer_dataset_path': "./data/referit3d/nr3d_test_sampled1000.csv",
        'use_gt_box': True,
        'use_priority': False,
        'use_code_interpreter': False,
        'use_object_filter': False,
    },
    17: {
        'dataset_type': 'nr3d',
        'model': 'gpt-3.5-turbo',  # 3.5 with principle, no code interpreter, no object filter
        'result_folder_name': 'eval_results_nr3d_35_p_no_code_no_filter_testset/',
        'use_principle': True,
        'refer_dataset_path': "./data/referit3d/nr3d_test_sampled1000.csv",
        'use_gt_box': True,
        'use_priority': False,
        'use_code_interpreter': False,
        'use_object_filter': False,
    },
    18: {
        'dataset_type': 'nr3d',
        'model': 'meta-llama/Llama-2-7b-chat-hf',
        'result_folder_name': 'eval_results_nr3d_llama2_7b_chat_p_no_filter_testset/',
        'use_principle': True,
        'refer_dataset_path': "./data/referit3d/nr3d_test_sampled1000.csv",
        'use_gt_box': True,
        'use_priority': False,
        'use_code_interpreter': False,
        'use_object_filter': False,
    },
    19: {
        'dataset_type': 'nr3d',
        'model': 'codellama/CodeLlama-7b-Instruct-hf',
        'result_folder_name': 'eval_results_nr3d_codellama_7b_instruct_p_no_filter_testset/',
        'use_principle': True,
        'refer_dataset_path': "./data/referit3d/nr3d_test_sampled1000.csv",
        'use_gt_box': True,
        'use_priority': False,
        'use_code_interpreter': False,
        'use_object_filter': False,
    },
    200: {
        'dataset_type': 'nr3d',
        'model': 'ft:gpt-3.5-turbo-1106:ripl::8LKEeL8m',  # finetuned model with principle, no code interpreter
        'result_folder_name': 'eval_results_nr3d_gpt35_refer_success_p_testset/',
        'use_principle': True,
        'refer_dataset_path': "./data/referit3d/nr3d_test_sampled1000.csv",
        'use_gt_box': True,
        'use_priority': False,
        'use_code_interpreter': False
    },
    201: {
        'dataset_type': 'nr3d',
        'model': 'ft:gpt-3.5-turbo-1106:ripl::8LKEeL8m',  # finetuned model no principle, no code interpreter
        'result_folder_name': 'eval_results_nr3d_gpt35_refer_success_np_testset/',
        'use_principle': False,
        'refer_dataset_path': "./data/referit3d/nr3d_test_sampled1000.csv",
        'use_gt_box': True,
        'use_priority': False,
        'use_code_interpreter': False
    },
    202: {
        'dataset_type': 'nr3d',
        'model': 'ft:gpt-3.5-turbo-1106:ripl::8LKnhl90',  # finetuned model with principle, no code interpreter
        'result_folder_name': 'eval_results_nr3d_gpt35_refer_success_correction_p_testset/',
        'use_principle': True,
        'refer_dataset_path': "./data/referit3d/nr3d_test_sampled1000.csv",
        'use_gt_box': True,
        'use_priority': False,
        'use_code_interpreter': False
    },
    203: {
        'dataset_type': 'nr3d',
        'model': 'ft:gpt-3.5-turbo-1106:ripl::8LKnhl90',  # finetuned model no principle, no code interpreter
        'result_folder_name': 'eval_results_nr3d_gpt35_refer_success_correction_np_testset/',
        'use_principle': False,
        'refer_dataset_path': "./data/referit3d/nr3d_test_sampled1000.csv",
        'use_gt_box': True,
        'use_priority': False,
        'use_code_interpreter': False
    },
    204: {
        'dataset_type': 'nr3d',
        'model': 'ft:gpt-3.5-turbo-1106:ripl::8LLmaMeG',  # finetuned model with principle, no code interpreter
        'result_folder_name': 'eval_results_nr3d_gpt35_no_rule_refer_success_p_testset/',
        'use_principle': True,
        'refer_dataset_path': "./data/referit3d/nr3d_test_sampled1000.csv",
        'use_gt_box': True,
        'use_priority': False,
        'use_code_interpreter': False
    },
    205: {
        'dataset_type': 'nr3d',
        'model': 'ft:gpt-3.5-turbo-1106:ripl::8LLmaMeG',  # finetuned model no principle, no code interpreter
        'result_folder_name': 'eval_results_nr3d_gpt35_no_rule_refer_success_np_testset/',
        'use_principle': False,
        'refer_dataset_path': "./data/referit3d/nr3d_test_sampled1000.csv",
        'use_gt_box': True,
        'use_priority': False,
        'use_code_interpreter': False
    },
    206: {
        'dataset_type': 'nr3d',
        'model': 'ft:gpt-3.5-turbo-1106:ripl::8LMeizps',  # finetuned model with principle, no code interpreter
        'result_folder_name': 'eval_results_nr3d_gpt35_no_rule_refer_success_correction_p_testset/',
        'use_principle': True,
        'refer_dataset_path': "./data/referit3d/nr3d_test_sampled1000.csv",
        'use_gt_box': True,
        'use_priority': False,
        'use_code_interpreter': False
    },
    207: {
        'dataset_type': 'nr3d',
        'model': 'ft:gpt-3.5-turbo-1106:ripl::8LMeizps',  # finetuned model no principle, no code interpreter
        'result_folder_name': 'eval_results_nr3d_gpt35_no_rule_refer_success_correction_np_testset/',
        'use_principle': False,
        'refer_dataset_path': "./data/referit3d/nr3d_test_sampled1000.csv",
        'use_gt_box': True,
        'use_priority': False,
        'use_code_interpreter': False,
    },
    210: {
        'dataset_type': 'nr3d',
        'model': 'checkpoints/refer_success-codellama-7b-finetune-pad-eos',  # finetuned model with principle, no code interpreter
        'result_folder_name': 'eval_results_nr3d_codellama_refer_success_p_testset/',
        'use_principle': True,
        'refer_dataset_path': "./data/referit3d/nr3d_test_sampled1000.csv",
        'use_gt_box': True,
        'use_priority': False,
        'use_code_interpreter': False
    },
    211: {
        'dataset_type': 'nr3d',
        'model': 'checkpoints/refer_success-codellama-7b-finetune-pad-eos',  # finetuned model no principle, no code interpreter
        'result_folder_name': 'eval_results_nr3d_codellama_refer_success_np_testset/',
        'use_principle': False,
        'refer_dataset_path': "./data/referit3d/nr3d_test_sampled1000.csv",
        'use_gt_box': True,
        'use_priority': False,
        'use_code_interpreter': False
    },
    212: {
        'dataset_type': 'nr3d',
        'model': 'checkpoints/refer_success_correction-codellama-7b-finetune-pad-eos',  # finetuned model with principle, no code interpreter
        'result_folder_name': 'eval_results_nr3d_codellama_refer_success_correction_p_testset/',
        'use_principle': True,
        'refer_dataset_path': "./data/referit3d/nr3d_test_sampled1000.csv",
        'use_gt_box': True,
        'use_priority': False,
        'use_code_interpreter': False
    },
    213: {
        'dataset_type': 'nr3d',
        'model': 'checkpoints/refer_success_correction-codellama-7b-finetune-pad-eos',  # finetuned model no principle, no code interpreter
        'result_folder_name': 'eval_results_nr3d_codellama_refer_success_correction_np_testset/',
        'use_principle': False,
        'refer_dataset_path': "./data/referit3d/nr3d_test_sampled1000.csv",
        'use_gt_box': True,
        'use_priority': False,
        'use_code_interpreter': False
    },
    214: {
        'dataset_type': 'nr3d',
        'model': 'checkpoints/no_rules_refer_success-codellama-7b-finetune-pad-eos',  # finetuned model with principle, no code interpreter
        'result_folder_name': 'eval_results_nr3d_codellama_no_rule_refer_success_p_testset/',
        'use_principle': True,
        'refer_dataset_path': "./data/referit3d/nr3d_test_sampled1000.csv",
        'use_gt_box': True,
        'use_priority': False,
        'use_code_interpreter': False
    },
    215: {
        'dataset_type': 'nr3d',
        'model': 'checkpoints/no_rules_refer_success-codellama-7b-finetune-pad-eos',  # finetuned model no principle, no code interpreter
        'result_folder_name': 'eval_results_nr3d_codellama_no_rule_refer_success_np_testset/',
        'use_principle': False,
        'refer_dataset_path': "./data/referit3d/nr3d_test_sampled1000.csv",
        'use_gt_box': True,
        'use_priority': False,
        'use_code_interpreter': False
    },
    216: {
        'dataset_type': 'nr3d',
        'model': 'checkpoints/no_rules_refer_success_correction-codellama-7b-finetune-pad-eos',  # finetuned model with principle, no code interpreter
        'result_folder_name': 'eval_results_nr3d_codellama_no_rule_refer_success_correction_p_testset/',
        'use_principle': True,
        'refer_dataset_path': "./data/referit3d/nr3d_test_sampled1000.csv",
        'use_gt_box': True,
        'use_priority': False,
        'use_code_interpreter': False
    },
    217: {
        'dataset_type': 'nr3d',
        'model': 'checkpoints/no_rules_refer_success_correction-codellama-7b-finetune-pad-eos',  # finetuned model no principle, no code interpreter
        'result_folder_name': 'eval_results_nr3d_codellama_no_rule_refer_success_correction_np_testset/',
        'use_principle': False,
        'refer_dataset_path': "./data/referit3d/nr3d_test_sampled1000.csv",
        'use_gt_box': True,
        'use_priority': False,
        'use_code_interpreter': False
    },
    220: {
        'dataset_type': 'nr3d',
        'model': 'checkpoints/refer_success-llama2-7b-chat-finetune-pad-eos',  # finetuned model with principle, no code interpreter
        'result_folder_name': 'eval_results_nr3d_llama2_refer_success_p_testset/',
        'use_principle': True,
        'refer_dataset_path': "./data/referit3d/nr3d_test_sampled1000.csv",
        'use_gt_box': True,
        'use_priority': False,
        'use_code_interpreter': False
    },
    221: {
        'dataset_type': 'nr3d',
        'model': 'checkpoints/refer_success-llama2-7b-chat-finetune-pad-eos',  # finetuned model no principle, no code interpreter
        'result_folder_name': 'eval_results_nr3d_llama2_refer_success_np_testset/',
        'use_principle': False,
        'refer_dataset_path': "./data/referit3d/nr3d_test_sampled1000.csv",
        'use_gt_box': True,
        'use_priority': False,
        'use_code_interpreter': False
    },
    222: {
        'dataset_type': 'nr3d',
        'model': 'checkpoints/refer_success_correction-llama2-7b-chat-finetune-pad-eos',  # finetuned model with principle, no code interpreter
        'result_folder_name': 'eval_results_nr3d_llama2_refer_success_correction_p_testset/',
        'use_principle': True,
        'refer_dataset_path': "./data/referit3d/nr3d_test_sampled1000.csv",
        'use_gt_box': True,
        'use_priority': False,
        'use_code_interpreter': False
    },
    223: {
        'dataset_type': 'nr3d',
        'model': 'checkpoints/refer_success_correction-llama2-7b-chat-finetune-pad-eos',  # finetuned model no principle, no code interpreter
        'result_folder_name': 'eval_results_nr3d_llama2_refer_success_correction_np_testset/',
        'use_principle': False,
        'refer_dataset_path': "./data/referit3d/nr3d_test_sampled1000.csv",
        'use_gt_box': True,
        'use_priority': False,
        'use_code_interpreter': False
    },
    224: {
        'dataset_type': 'nr3d',
        'model': 'checkpoints/no_rules_refer_success-llama2-7b-chat-finetune-pad-eos',  # finetuned model with principle, no code interpreter
        'result_folder_name': 'eval_results_nr3d_llama2_no_rule_refer_success_p_testset/',
        'use_principle': True,
        'refer_dataset_path': "./data/referit3d/nr3d_test_sampled1000.csv",
        'use_gt_box': True,
        'use_priority': False,
        'use_code_interpreter': False
    },
    225: {
        'dataset_type': 'nr3d',
        'model': 'checkpoints/no_rules_refer_success-llama2-7b-chat-finetune-pad-eos',  # finetuned model no principle, no code interpreter
        'result_folder_name': 'eval_results_nr3d_llama2_no_rule_refer_success_np_testset/',
        'use_principle': False,
        'refer_dataset_path': "./data/referit3d/nr3d_test_sampled1000.csv",
        'use_gt_box': True,
        'use_priority': False,
        'use_code_interpreter': False
    },
    226: {
        'dataset_type': 'nr3d',
        'model': 'checkpoints/no_rules_refer_success_correction-llama2-7b-chat-finetune-pad-eos',  # finetuned model with principle, no code interpreter
        'result_folder_name': 'eval_results_nr3d_llama2_no_rule_refer_success_correction_p_testset/',
        'use_principle': True,
        'refer_dataset_path': "./data/referit3d/nr3d_test_sampled1000.csv",
        'use_gt_box': True,
        'use_priority': False,
        'use_code_interpreter': False
    },
    227: {
        'dataset_type': 'nr3d',
        'model': 'checkpoints/no_rules_refer_success_correction-llama2-7b-chat-finetune-pad-eos',  # finetuned model no principle, no code interpreter
        'result_folder_name': 'eval_results_nr3d_llama2_no_rule_refer_success_correction_np_testset/',
        'use_principle': False,
        'refer_dataset_path': "./data/referit3d/nr3d_test_sampled1000.csv",
        'use_gt_box': True,
        'use_priority': False,
        'use_code_interpreter': False
    },
    300: {
        'dataset_type': 'nr3d',
        'model': 'ft:gpt-3.5-turbo-1106:ripl::8LKEeL8m',  # finetuned model with principle, with code interpreter
        'result_folder_name': 'eval_results_nr3d_gpt35_refer_success_p_code_testset/',
        'use_principle': True,
        'refer_dataset_path': "./data/referit3d/nr3d_test_sampled1000.csv",
        'use_gt_box': True,
        'use_priority': False,
        'use_code_interpreter': True
    },
    301: {
        'dataset_type': 'nr3d',
        'model': 'ft:gpt-3.5-turbo-1106:ripl::8LKEeL8m',  # finetuned model no principle, with code interpreter
        'result_folder_name': 'eval_results_nr3d_gpt35_refer_success_np_code_testset/',
        'use_principle': False,
        'refer_dataset_path': "./data/referit3d/nr3d_test_sampled1000.csv",
        'use_gt_box': True,
        'use_priority': False,
        'use_code_interpreter': True
    },
    302: {
        'dataset_type': 'nr3d',
        'model': 'ft:gpt-3.5-turbo-1106:ripl::8LKnhl90',  # finetuned model with principle, with code interpreter
        'result_folder_name': 'eval_results_nr3d_gpt35_refer_success_correction_p_code_testset/',
        'use_principle': True,
        'refer_dataset_path': "./data/referit3d/nr3d_test_sampled1000.csv",
        'use_gt_box': True,
        'use_priority': False,
        'use_code_interpreter': True
    },
    303: {
        'dataset_type': 'nr3d',
        'model': 'ft:gpt-3.5-turbo-1106:ripl::8LKnhl90',  # finetuned model no principle, with code interpreter
        'result_folder_name': 'eval_results_nr3d_gpt35_refer_success_correction_np_code_testset/',
        'use_principle': False,
        'refer_dataset_path': "./data/referit3d/nr3d_test_sampled1000.csv",
        'use_gt_box': True,
        'use_priority': False,
        'use_code_interpreter': True
    },
    304: {
        'dataset_type': 'nr3d',
        'model': 'ft:gpt-3.5-turbo-1106:ripl::8LLmaMeG',  # finetuned model with principle, with code interpreter
        'result_folder_name': 'eval_results_nr3d_gpt35_no_rule_refer_success_p_code_testset/',
        'use_principle': True,
        'refer_dataset_path': "./data/referit3d/nr3d_test_sampled1000.csv",
        'use_gt_box': True,
        'use_priority': False,
        'use_code_interpreter': True
    },
    305: {
        'dataset_type': 'nr3d',
        'model': 'ft:gpt-3.5-turbo-1106:ripl::8LLmaMeG',  # finetuned model no principle, with code interpreter
        'result_folder_name': 'eval_results_nr3d_gpt35_no_rule_refer_success_np_code_testset/',
        'use_principle': False,
        'refer_dataset_path': "./data/referit3d/nr3d_test_sampled1000.csv",
        'use_gt_box': True,
        'use_priority': False,
        'use_code_interpreter': True
    },
    306: {
        'dataset_type': 'nr3d',
        'model': 'ft:gpt-3.5-turbo-1106:ripl::8LMeizps',  # finetuned model with principle, with code interpreter
        'result_folder_name': 'eval_results_nr3d_gpt35_no_rule_refer_success_correction_p_code_testset/',
        'use_principle': True,
        'refer_dataset_path': "./data/referit3d/nr3d_test_sampled1000.csv",
        'use_gt_box': True,
        'use_priority': False,
        'use_code_interpreter': True
    },
    307: {
        'dataset_type': 'nr3d',
        'model': 'ft:gpt-3.5-turbo-1106:ripl::8LMeizps',  # finetuned model no principle, with code interpreter
        'result_folder_name': 'eval_results_nr3d_gpt35_no_rule_refer_success_correction_np_code_testset/',
        'use_principle': False,
        'refer_dataset_path': "./data/referit3d/nr3d_test_sampled1000.csv",
        'use_gt_box': True,
        'use_priority': False,
        'use_code_interpreter': True,
    },
    310: {
        'dataset_type': 'nr3d',
        'model': 'checkpoints/refer_success-codellama-7b-finetune-pad-eos',  # finetuned model with principle, with code interpreter
        'result_folder_name': 'eval_results_nr3d_codellama_refer_success_p_code_testset/',
        'use_principle': True,
        'refer_dataset_path': "./data/referit3d/nr3d_test_sampled1000.csv",
        'use_gt_box': True,
        'use_priority': False,
        'use_code_interpreter': True
    },
    311: {
        'dataset_type': 'nr3d',
        'model': 'checkpoints/refer_success-codellama-7b-finetune-pad-eos',  # finetuned model no principle, with code interpreter
        'result_folder_name': 'eval_results_nr3d_codellama_refer_success_np_code_testset/',
        'use_principle': False,
        'refer_dataset_path': "./data/referit3d/nr3d_test_sampled1000.csv",
        'use_gt_box': True,
        'use_priority': False,
        'use_code_interpreter': True
    },
    312: {
        'dataset_type': 'nr3d',
        'model': 'checkpoints/refer_success_correction-codellama-7b-finetune-pad-eos',  # finetuned model with principle, no code interpreter
        'result_folder_name': 'eval_results_nr3d_codellama_refer_success_correction_p_code_testset/',
        'use_principle': True,
        'refer_dataset_path': "./data/referit3d/nr3d_test_sampled1000.csv",
        'use_gt_box': True,
        'use_priority': False,
        'use_code_interpreter': True
    },
    313: {
        'dataset_type': 'nr3d',
        'model': 'checkpoints/refer_success_correction-codellama-7b-finetune-pad-eos',  # finetuned model no principle, no code interpreter
        'result_folder_name': 'eval_results_nr3d_codellama_refer_success_correction_np_code_testset/',
        'use_principle': False,
        'refer_dataset_path': "./data/referit3d/nr3d_test_sampled1000.csv",
        'use_gt_box': True,
        'use_priority': False,
        'use_code_interpreter': True
    },
    314: {
        'dataset_type': 'nr3d',
        'model': 'checkpoints/no_rules_refer_success-codellama-7b-finetune-pad-eos',  # finetuned model with principle, no code interpreter
        'result_folder_name': 'eval_results_nr3d_codellama_no_rule_refer_success_p_code_testset/',
        'use_principle': True,
        'refer_dataset_path': "./data/referit3d/nr3d_test_sampled1000.csv",
        'use_gt_box': True,
        'use_priority': False,
        'use_code_interpreter': True
    },
    315: {
        'dataset_type': 'nr3d',
        'model': 'checkpoints/no_rules_refer_success-codellama-7b-finetune-pad-eos',  # finetuned model no principle, no code interpreter
        'result_folder_name': 'eval_results_nr3d_codellama_no_rule_refer_success_np_code_testset/',
        'use_principle': False,
        'refer_dataset_path': "./data/referit3d/nr3d_test_sampled1000.csv",
        'use_gt_box': True,
        'use_priority': False,
        'use_code_interpreter': True
    },
    316: {
        'dataset_type': 'nr3d',
        'model': 'checkpoints/no_rules_refer_success_correction-codellama-7b-finetune-pad-eos',  # finetuned model with principle, no code interpreter
        'result_folder_name': 'eval_results_nr3d_codellama_no_rule_refer_success_correction_p_code_testset/',
        'use_principle': True,
        'refer_dataset_path': "./data/referit3d/nr3d_test_sampled1000.csv",
        'use_gt_box': True,
        'use_priority': False,
        'use_code_interpreter': True
    },
    317: {
        'dataset_type': 'nr3d',
        'model': 'checkpoints/no_rules_refer_success_correction-codellama-7b-finetune-pad-eos',  # finetuned model no principle, with code interpreter
        'result_folder_name': 'eval_results_nr3d_codellama_no_rule_refer_success_correction_np_code_testset/',
        'use_principle': False,
        'refer_dataset_path': "./data/referit3d/nr3d_test_sampled1000.csv",
        'use_gt_box': True,
        'use_priority': False,
        'use_code_interpreter': True
    },
    320: {
        'dataset_type': 'nr3d',
        'model': 'checkpoints/refer_success-llama2-7b-chat-finetune-pad-eos',  # finetuned model with principle, with code interpreter
        'result_folder_name': 'eval_results_nr3d_llama2_refer_success_p_code_testset/',
        'use_principle': True,
        'refer_dataset_path': "./data/referit3d/nr3d_test_sampled1000.csv",
        'use_gt_box': True,
        'use_priority': False,
        'use_code_interpreter': True
    },
    321: {
        'dataset_type': 'nr3d',
        'model': 'checkpoints/refer_success-llama2-7b-chat-finetune-pad-eos',  # finetuned model no principle, no code interpreter
        'result_folder_name': 'eval_results_nr3d_llama2_refer_success_np_code_testset/',
        'use_principle': False,
        'refer_dataset_path': "./data/referit3d/nr3d_test_sampled1000.csv",
        'use_gt_box': True,
        'use_priority': False,
        'use_code_interpreter': True
    },
    322: {
        'dataset_type': 'nr3d',
        'model': 'checkpoints/refer_success_correction-llama2-7b-chat-finetune-pad-eos',  # finetuned model with principle, no code interpreter
        'result_folder_name': 'eval_results_nr3d_llama2_refer_success_correction_p_code_testset/',
        'use_principle': True,
        'refer_dataset_path': "./data/referit3d/nr3d_test_sampled1000.csv",
        'use_gt_box': True,
        'use_priority': False,
        'use_code_interpreter': True
    },
    323: {
        'dataset_type': 'nr3d',
        'model': 'checkpoints/refer_success_correction-llama2-7b-chat-finetune-pad-eos',  # finetuned model no principle, no code interpreter
        'result_folder_name': 'eval_results_nr3d_llama2_refer_success_correction_np_code_testset/',
        'use_principle': False,
        'refer_dataset_path': "./data/referit3d/nr3d_test_sampled1000.csv",
        'use_gt_box': True,
        'use_priority': False,
        'use_code_interpreter': True
    },
    324: {
        'dataset_type': 'nr3d',
        'model': 'checkpoints/no_rules_refer_success-llama2-7b-chat-finetune-pad-eos',  # finetuned model with principle, no code interpreter
        'result_folder_name': 'eval_results_nr3d_llama2_no_rule_refer_success_p_code_testset/',
        'use_principle': True,
        'refer_dataset_path': "./data/referit3d/nr3d_test_sampled1000.csv",
        'use_gt_box': True,
        'use_priority': False,
        'use_code_interpreter': True
    },
    325: {
        'dataset_type': 'nr3d',
        'model': 'checkpoints/no_rules_refer_success-llama2-7b-chat-finetune-pad-eos',  # finetuned model no principle, no code interpreter
        'result_folder_name': 'eval_results_nr3d_llama2_no_rule_refer_success_np_code_testset/',
        'use_principle': False,
        'refer_dataset_path': "./data/referit3d/nr3d_test_sampled1000.csv",
        'use_gt_box': True,
        'use_priority': False,
        'use_code_interpreter': True
    },
    326: {
        'dataset_type': 'nr3d',
        'model': 'checkpoints/no_rules_refer_success_correction-llama2-7b-chat-finetune-pad-eos',  # finetuned model with principle, no code interpreter
        'result_folder_name': 'eval_results_nr3d_llama2_no_rule_refer_success_correction_p_code_testset/',
        'use_principle': True,
        'refer_dataset_path': "./data/referit3d/nr3d_test_sampled1000.csv",
        'use_gt_box': True,
        'use_priority': False,
        'use_code_interpreter': True
    },
    327: {
        'dataset_type': 'nr3d',
        'model': 'checkpoints/no_rules_refer_success_correction-llama2-7b-chat-finetune-pad-eos',  # finetuned model no principle, no code interpreter
        'result_folder_name': 'eval_results_nr3d_llama2_no_rule_refer_success_correction_np_code_testset/',
        'use_principle': False,
        'refer_dataset_path': "./data/referit3d/nr3d_test_sampled1000.csv",
        'use_gt_box': True,
        'use_priority': False,
        'use_code_interpreter': True
    },
}

confs_sr3d = {
    # sr3d
    0: {
        'dataset_type': 'sr3d',
        'model': 'gpt-4',  # 4 no principle
        'result_folder_name': 'eval_results_sr3d_4_np_testset/',
        'use_principle': False,
        'refer_dataset_path': "./data/referit3d/sr3d_test_sampled1000.csv",
        'use_gt_box': True,
        'use_priority': False,
        'use_code_interpreter': True
    },
    1: {
        'dataset_type': 'sr3d',
        'model': 'gpt-4',  # 4 with principle
        'result_folder_name': 'eval_results_sr3d_4_p_testset/',
        'use_principle': True,
        'refer_dataset_path': "./data/referit3d/sr3d_test_sampled1000.csv",
        'use_gt_box': True,
        'use_priority': False,
        'use_code_interpreter': True
    },
    100: {
        'dataset_type': 'sr3d',
        'model': 'gpt-4-1106-preview',  # 4 no principle
        'result_folder_name': 'eval_results_sr3d_4tb_np_testset/',
        'use_principle': False,
        'refer_dataset_path': "./data/referit3d/sr3d_test_sampled1000.csv",
        'use_gt_box': True,
        'use_priority': False,
        'use_code_interpreter': True
    },
    101: {
        'dataset_type': 'sr3d',
        'model': 'gpt-4-1106-preview',  # 4 with principle
        'result_folder_name': 'eval_results_sr3d_4tb_p_testset/',
        'use_principle': True,
        'refer_dataset_path': "./data/referit3d/sr3d_test_sampled1000.csv",
        'use_gt_box': True,
        'use_priority': False,
        'use_code_interpreter': True
    },
    2: {
        'dataset_type': 'sr3d',
        'model': 'gpt-3.5-turbo',  # 3.5 zero shot no principle
        'result_folder_name': 'eval_results_sr3d_35_np_testset/',
        'use_principle': False,
        'refer_dataset_path': "./data/referit3d/sr3d_test_sampled1000.csv",
        'use_gt_box': True,
        'use_priority': False,
        'use_code_interpreter': True
    },
    3: {
        'dataset_type': 'sr3d',
        'model': 'gpt-3.5-turbo',  # 3.5 zero shot with principle
        'result_folder_name': 'eval_results_sr3d_35_p_testset/',
        'use_principle': True,
        'refer_dataset_path': "./data/referit3d/sr3d_test_sampled1000.csv",
        'use_gt_box': True,
        'use_priority': False,
        'use_code_interpreter': True
    },
    4: {
        'dataset_type': 'sr3d',
        'model': 'ft:gpt-3.5-turbo-0613:ripl:ref-suc573-cor233:7z8ztnAu',  # 3.5 ft suc573加cor233
        'result_folder_name': 'eval_results_sr3d_35_np_sc_testset/',
        'use_principle': False,
        'refer_dataset_path': "./data/referit3d/sr3d_test_sampled1000.csv",
        'use_gt_box': True,
        'use_priority': False,
        'use_code_interpreter': True
    },
    5: {
        'dataset_type': 'sr3d',
        'model': 'ft:gpt-3.5-turbo-0613:ripl:refer-succ-166-nr:7yp0M6Nn',  # 3.5 ft success573
        'result_folder_name': 'eval_results_sr3d_35_np_c_testset/',
        'use_principle': False,
        'refer_dataset_path': "./data/referit3d/sr3d_test_sampled1000.csv",
        'use_gt_box': True,
        'use_priority': False,
        'use_code_interpreter': True
    },
    # sr3d
    6: {
        'dataset_type': 'sr3d',
        'model': 'gpt-4',  # sr3d no code interpreter, no principle
        'result_folder_name': 'eval_results_sr3d_4_np_as_nocode_testset/',
        'use_principle': False,
        'refer_dataset_path': "./data/referit3d/sr3d_test_assembled30x5.csv",
        'use_gt_box': True,
        'use_priority': False,
        'use_code_interpreter': False
    },
    7: {
        'dataset_type': 'sr3d',
        'model': 'gpt-4',  # sr3d with code interpreter, no principle
        'result_folder_name': 'eval_results_sr3d_4_np_as_code_testset/',
        'use_principle': False,
        'refer_dataset_path': "./data/referit3d/sr3d_test_assembled30x5.csv",
        'use_gt_box': True,
        'use_priority': False,
        'use_code_interpreter': True
    },
    8: {
        'dataset_type': 'sr3d',
        'model': 'gpt-4',  # sr3d no code interpreter, with principle
        'result_folder_name': 'eval_results_sr3d_4_p_as_nocode_testset/',
        'use_principle': True,
        'refer_dataset_path': "./data/referit3d/sr3d_test_assembled30x5.csv",
        'use_gt_box': True,
        'use_priority': False,
        'use_code_interpreter': False
    },
    9: {
        'dataset_type': 'sr3d',
        'model': 'gpt-4',  # sr3d with code interpreter, with principle
        'result_folder_name': 'eval_results_sr3d_4_p_as_code_testset/',
        'use_principle': True,
        'refer_dataset_path': "./data/referit3d/sr3d_test_assembled30x5.csv",
        'use_gt_box': True,
        'use_priority': False,
        'use_code_interpreter': True
    },
    10: {
        'dataset_type': 'sr3d',
        'model': 'meta-llama/Llama-2-7b-chat-hf',  # llama2-7b-chat with principle
        'result_folder_name': 'eval_results_sr3d_llama2_7b_chat_p_testset/',
        'use_principle': True,
        'refer_dataset_path': "./data/referit3d/sr3d_test_sampled1000.csv",
        'use_gt_box': True,
        'use_priority': False,
        'use_code_interpreter': False
    },
    11: {
        'dataset_type': 'sr3d',
        'model': 'codellama/CodeLlama-7b-Instruct-hf',  # llama2-7b-chat with principle
        'result_folder_name': 'eval_results_sr3d_codellama_7b_instruct_p_testset/',
        'use_principle': True,
        'refer_dataset_path': "./data/referit3d/sr3d_test_sampled1000.csv",
        'use_gt_box': True,
        'use_priority': False,
        'use_code_interpreter': False
    },
    12: {
        'dataset_type': 'sr3d',
        'model': 'meta-llama/Llama-2-7b-chat-hf',  # llama2-7b-chat with principle
        'result_folder_name': 'eval_results_sr3d_llama2_7b_chat_p_no_filter_testset/',
        'use_principle': True,
        'refer_dataset_path': "./data/referit3d/sr3d_test_sampled1000.csv",
        'use_gt_box': True,
        'use_priority': False,
        'use_code_interpreter': False,
        'use_object_filter': False,
    },
    13: {
        'dataset_type': 'sr3d',
        'model': 'codellama/CodeLlama-7b-Instruct-hf',  # llama2-7b-chat with principle
        'result_folder_name': 'eval_results_sr3d_codellama_7b_instruct_p_no_filter_testset/',
        'use_principle': True,
        'refer_dataset_path': "./data/referit3d/sr3d_test_sampled1000.csv",
        'use_gt_box': True,
        'use_priority': False,
        'use_code_interpreter': False,
        'use_object_filter': False,
    },
    14: {
        'dataset_type': 'sr3d',
        'model': 'gpt-3.5-turbo',  # 3.5 zero shot principle
        'result_folder_name': 'eval_results_sr3d_35_p_no_code_testset/',
        'use_principle': True,
        'refer_dataset_path': "./data/referit3d/sr3d_test_sampled1000.csv",
        'use_gt_box': True,
        'use_priority': False,
        'use_code_interpreter': False
    },
    15: {
        'dataset_type': 'sr3d',
        'model': 'gpt-4',  # sr3d no code interpreter, with principle
        'result_folder_name': 'eval_results_sr3d_4_p_no_code_testset/',
        'use_principle': True,
        'refer_dataset_path': "./data/referit3d/sr3d_test_sampled1000.csv",
        'use_gt_box': True,
        'use_priority': False,
        'use_code_interpreter': False
    },
    # 3.5-turbo, sampled 1000, no CI
    16: {
        'dataset_type': 'sr3d',
        'model': 'gpt-3.5-turbo',
        'result_folder_name': 'eval_results_sr3d_35_p_testset/',
        'use_principle': True,
        'refer_dataset_path': "./data/referit3d/sr3d_test_sampled1000.csv",
        'use_gt_box': True,
        'use_priority': False,
        'use_code_interpreter': False
    },
    # 3.5-turbo, assembled 100x5, CI
    17: {
        'dataset_type': 'sr3d',
        'model': 'gpt-3.5-turbo',
        'result_folder_name': 'eval_results_sr3d_35_p_test_assembled/',
        'use_principle': True,
        'refer_dataset_path': "./data/referit3d/sr3d_test_assembled100x5.csv",
        'use_gt_box': True,
        'use_priority': False,
        'use_code_interpreter': True
    },
    # 3.5-turbo, assembled 100x5, no CI
    18: {
        'dataset_type': 'sr3d',
        'model': 'gpt-3.5-turbo',
        'result_folder_name': 'eval_results_sr3d_35_p_test_assembled/',
        'use_principle': True,
        'refer_dataset_path': "./data/referit3d/sr3d_test_assembled100x5.csv",
        'use_gt_box': True,
        'use_priority': False,
        'use_code_interpreter': False
    },
    # 4-turbo, assembled 100x5, no CI
    19: {
        'dataset_type': 'sr3d',
        'model': 'gpt-4-1106-preview',
        'result_folder_name': 'eval_results_sr3d_4_p_test_assembled/',
        'use_principle': True,
        'refer_dataset_path': "./data/referit3d/sr3d_test_assembled100x5.csv",
        'use_gt_box': True,
        'use_priority': False,
        'use_code_interpreter': False
    },
}

confs_scanrefer = {
    # scanrefer
    0: {
        'dataset_type': 'scanrefer',
        'model': 'gpt-4',  # 4 with gt box on val set
        'result_folder_name': 'eval_results_scanrefer_4_p_gtbox_valset/',
        'use_principle': True,
        'refer_dataset_path': "./data/scanrefer/ScanRefer_filtered_val_sampled1000.json",
        'use_gt_box': True,
        'use_priority': False,
        'use_code_interpreter': True
    },
    100: {
        'dataset_type': 'scanrefer',
        'model': 'gpt-4-1106-preview',  # 4 turbo with gt box on val set
        'result_folder_name': 'eval_results_scanrefer_4tb_p_gtbox_valset/',
        'use_principle': True,
        'refer_dataset_path': "./data/scanrefer/ScanRefer_filtered_val_sampled1000.json",
        'use_gt_box': True,
        'use_priority': False,
        'use_code_interpreter': True
    },
    1: {
        'dataset_type': 'scanrefer',
        'model': 'gpt-4',  # 4 with gt box on train set
        'result_folder_name': 'eval_results_scanrefer_2stages/',
        'use_principle': True,
        'refer_dataset_path': "./data/scanrefer/ScanRefer_filtered_train_sampled1000.json",
        'use_gt_box': True,
        'use_priority': False,
        'use_code_interpreter': True
    },
    2: {
        'dataset_type': 'scanrefer',
        'model': 'gpt-4',  # 4 with mask3d 200c on val set
        'result_folder_name': 'eval_results_scanrefer_4_p_mask3d_200c_valset/',
        'use_principle': True,
        'refer_dataset_path': "./data/scanrefer/ScanRefer_filtered_val_sampled1000.json",
        'use_gt_box': False,
        'use_priority': False,
        'use_code_interpreter': True,
        'tool': 'mask3d_200c'
    },
    3: {
        'dataset_type': 'scanrefer',
        'model': 'gpt-4',  # 4 with mask3d 20c on val set
        'result_folder_name': 'eval_results_scanrefer_4_p_mask3d_20c_valset/',
        'use_principle': True,
        'refer_dataset_path': "./data/scanrefer/ScanRefer_filtered_val_sampled1000.json",
        'use_gt_box': False,
        'use_priority': False,
        'use_code_interpreter': True,
        'tool': 'mask3d_20c'
    },
    4: {
        'dataset_type': 'scanrefer',
        'model': 'gpt-4',  # 4 with gf val set
        'result_folder_name': 'eval_results_scanrefer_4_p_gf_valset/',
        'use_principle': True,
        'refer_dataset_path': "./data/scanrefer/ScanRefer_filtered_val_sampled1000.json",  # 改正
        'use_gt_box': False,
        'use_priority': False,
        'use_code_interpreter': True,
        'tool': 'gf'
    },
    5: {
        'dataset_type': 'scanrefer',
        'model': 'gpt-4',  # 4 with gf all sets sample 50
        'result_folder_name': 'eval_results_scanrefer_2stages/',
        'use_principle': True,
        'refer_dataset_path': "./data/scanrefer/ScanRefer_filtered_sampled50.json",
        'use_gt_box': False,
        'use_priority': False,
        'use_code_interpreter': True,
        'tool': 'gf'
    },
    6: {
        'dataset_type': 'scanrefer',
        'model': 'gpt-4',  # 4 with gf priority
        'result_folder_name': 'eval_results_scanrefer_4_p_gf_prio_valset/',
        'use_principle': True,
        'refer_dataset_path': "./data/scanrefer/ScanRefer_filtered_val_sampled1000.json",
        'use_gt_box': False,
        'use_priority': True,
        'use_code_interpreter': True,
        'tool': 'gf'
    },
    6: {
        'dataset_type': 'scanrefer',
        'model': 'gpt-4',  # 4 with 3dvista
        'result_folder_name': 'eval_results_scanrefer_4_p_3dvista_valset/',
        'use_principle': True,
        'refer_dataset_path': "./data/scanrefer/ScanRefer_filtered_val_sampled1000.json",
        'use_gt_box': False,
        'use_priority': False,
        'use_code_interpreter': True,
        'tool': '3dvista_200c'
    },
    7: {
        'dataset_type': 'scanrefer',
        'model': 'gpt-4',  # 4 with 3dvista without filtering pc
        # 'model':'gpt-4-1106-preview', # 4 with 3dvista without filtering pc
        'result_folder_name': 'eval_results_scanrefer_4_p_3dvista_valset/',
        'use_principle': True,
        'refer_dataset_path': "./data/scanrefer/ScanRefer_filtered_val_sampled1000.json",
        'use_gt_box': False,
        'use_priority': False,
        'use_code_interpreter': True,
        'tool': '3dvista_200c_wof'
    },
    107: {
        'dataset_type': 'scanrefer',
        # 'model':'gpt-4', # 4 with 3dvista without filtering pc
        'model': 'gpt-4-1106-preview',  # 4-turbo with 3dvista without filtering pc
        'result_folder_name': 'eval_results_scanrefer_4tb_p_3dvista_valset/',
        'use_principle': True,
        'refer_dataset_path': "./data/scanrefer/ScanRefer_filtered_val_sampled1000.json",
        'use_gt_box': False,
        'use_priority': False,
        'use_code_interpreter': True,
        'tool': '3dvista_200c_wof'
    },
    8: {
        'dataset_type': 'scanrefer',
        'model': 'gpt-4',  # full eval on gpt-4 with detected box
        'result_folder_name': 'eval_results_scanrefer_4_p_3dvista_valset/',
        'use_principle': True,
        'refer_dataset_path': "./data/scanrefer/ScanRefer_filtered_val.json",
        'use_gt_box': False,
        'use_priority': False,
        'use_code_interpreter': True,
        'tool': '3dvista_200c_wof'
    },
    108: {
        'dataset_type': 'scanrefer',
        'model': 'gpt-4-1106-preview',  # full eval on gpt-4-turbo with detected box
        'result_folder_name': 'eval_results_scanrefer_4tb_p_3dvista_valset/',
        'use_principle': True,
        'refer_dataset_path': "./data/scanrefer/ScanRefer_filtered_val.json",
        'use_gt_box': False,
        'use_priority': False,
        'use_code_interpreter': True,
        'tool': '3dvista_200c_wof'
    },
    9: {
        'dataset_type': 'scanrefer',
        'model': 'gpt-4',  # full eval on gpt-4 with gt box
        'result_folder_name': 'eval_results_scanrefer_4tb_p_gtbox_valset/',
        'use_principle': True,
        'refer_dataset_path': "./data/scanrefer/ScanRefer_filtered_val.json",
        'use_gt_box': False,
        'use_priority': False,
        'use_code_interpreter': True,
    },
    109: {
        'dataset_type': 'scanrefer',
        'model': 'gpt-4-1106-preview',  # full eval on gpt-4-turbo with gt box
        'result_folder_name': 'eval_results_scanrefer_4tb_p_gtbox_valset/',
        'use_principle': True,
        'refer_dataset_path': "./data/scanrefer/ScanRefer_filtered_val.json",
        'use_gt_box': False,
        'use_priority': False,
        'use_code_interpreter': True,
    },
}
