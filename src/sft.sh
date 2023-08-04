deepspeed  train.py --deepspeed_config ./configs/ds_config_zero2.json \
    --num_epochs 20 \
    --lora_r 16 \
    --train_file_path /media/E/lichunyu/datasets/chatglm/train.jsonl \
    --eval_file_path /media/E/lichunyu/datasets/chatglm/test.jsonl \
    --train_batch_size_per_gpu 8 \
    --eval_batch_size_per_gpu 8 \