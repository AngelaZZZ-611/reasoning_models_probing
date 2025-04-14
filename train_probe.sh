#!/bin/bash

MODEL=DeepSeek-R1-Distill-Qwen-1.5B
DATA=math-train
TRAIN_DATA_PATH=./model_embeds/${MODEL}_${DATA}

# Grid search over hyperparameters
for lr in 1e-3 1e-4 1e-5; do
    for hidden_size in 0 16 32; do
        for wd in 0.001 0.01 0.1; do
            for alpha in 0.5 0.7 0.9 1.0 1.5 2.0 3.0; do
                echo "Running with --lr $lr --hidden_size $hidden_size --wd $wd --alpha_imbalance_penalty $alpha"
                python -u ./train_predictor_with_class_weights.py \
                    --batch_size 64 \
                    --epochs 200 \
                    --lr "$lr" \
                    --hidden_size "$hidden_size" \
                    --wd "$wd" \
                    --threshold 0.5 \
                    --alpha_imbalance_penalty "$alpha" \
                    --train_data_dir $TRAIN_DATA_PATH \
                    --save_model_path ./grid_search/${MODEL}_${DATA}/checkpoints \
                    --store_path ./grid_search/${MODEL}_${DATA}/store \
                    --model_name $MODEL
            done
        done
    done
done
done
done