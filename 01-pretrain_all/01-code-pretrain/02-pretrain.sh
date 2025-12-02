#!/bin/bash

export TF_ENABLE_ONEDNN_OPTS=0


start_time=$(date +%s)

env CUDA_VISIBLE_DEVICES=0 python 01-pretrain.py  \
    --model_name_or_path  ../02-model_pretrain_example/bert-6MER-retokenizer \
    --train_data ../03-data_pretrain_example/pretrain_data.txt \
    --output_dir output_dir


end_time=$(date +%s)
echo "Total runtime: $((end_time - start_time)) seconds"
#Total runtime: 75 seconds

