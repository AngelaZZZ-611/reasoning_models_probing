#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

INPUT_SIZE=1536
INPUT_SIZE=4096
INPUT_SIZE=5120
INPUT_SIZE=3584
MODEL=DeepSeek-R1-Distill-Qwen-7B
# DATA=aime_83_24_by_batch
for MODEL in DeepSeek-R1-Distill-Llama-70B
do
# for DATA in gsm8k-train_by_batch math-train_by_batch 
for DATA in knowlogic-train_by_batch aime_83_24_by_batch
do
TRAIN_DATA_PATH=/home/test/test05/cyl/DS_reasoning_probing/DS_reasoning_probing/model_embeds/${MODEL}_${DATA}
# TEST_DATA_PATH=/home/test/test05/cyl/DS_reasoning_probing/DS_reasoning_probing/model_embeds/DeepSeek-R1-Distill-Qwen-7B_aime_25_by_batch
# Grid search over hyperparameters
for lr in 1e-3 1e-4 1e-5; do
    for hidden_size in 0 16 32; do
        for wd in 0.001 0.01 0.1; do
            for alpha in 0.5 0.7 0.9 1.0 1.5 2.0 3.0; do
                echo "Running with --lr $lr --hidden_size $hidden_size --wd $wd --alpha_imbalance_penalty $alpha"
                python -u ./train_predictor_with_class_weights.py \
                    --input_size $INPUT_SIZE \
                    --batch_size 64 \
                    --epochs 200 \
                    --lr "$lr" \
                    --hidden_size "$hidden_size" \
                    --wd "$wd" \
                    --threshold 0.5 \
                    --alpha_imbalance_penalty "$alpha" \
                    --train_data_dir $TRAIN_DATA_PATH \
                    --save_model_path /home/test/test05/cyl/DS_reasoning_probing/DS_reasoning_probing/grid_search/${MODEL}_${DATA}/checkpoints \
                    --store_path /home/test/test05/cyl/DS_reasoning_probing/DS_reasoning_probing/grid_search/${MODEL}_${DATA}/store \
                    --model_name $MODEL
            done
        done
    done
done
done
done