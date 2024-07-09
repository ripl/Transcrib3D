python trl_finetune.py \
    --model_name meta-llama/Llama-2-7b-chat-hf \
    --train_datapath finetune/finetune_files/all_refer_success_train.jsonl \
    --val_datapath finetune/finetune_files/all_refer_success_val.jsonl \
    --load_in_8bit \
    --use_peft \
    --batch_size 4 \
    --gradient_accumulation_steps 2 \
    --output_dir checkpoints/all_refer_success-llama2-7b-chat-finetune-pad-eos