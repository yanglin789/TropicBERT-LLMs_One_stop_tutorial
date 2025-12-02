#!/bin/bash

export TF_ENABLE_ONEDNN_OPTS=0


start_time=$(date +%s)

# regression\ classification\ multi-classification

env CUDA_VISIBLE_DEVICES=0 python 01-finetune.py  \
  --model_name_or_path  ../02-model_finetune_example/pretrained_model-bert-6MER-retokenizer-checkpoint-40/ \
  --train_task 'regression' \
  --reinit_classifier_layer True \
  --train_data ../03-data_finetune_example/regression_train_data.csv \
  --eval_data ../03-data_finetune_example/regression_dev_data.csv  \
  --test_data ../03-data_finetune_example/regression_test_data.csv \
  --output_dir output_fintue   \
  --model_max_length 512  \
  --run_nam runs_fintue  \
  --gradient_accumulation_steps 4 \
  --per_device_train_batch_size 16  \
  --per_device_eval_batch_size 16 \
  --num_train_epochs 5  \
  --logging_steps 50  \
  --eval_steps  50  \
  --save_steps  1000 \
  --warmup_ratio  0.05  \
  --weight_decay  0.01  \
  --learning_rate 1e-5  \
  --save_total_limit 5

end_time=$(date +%s)
echo "Total runtime: $((end_time - start_time)) seconds"
# Total runtime: 85 seconds



