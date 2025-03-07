#!/bin/bash

# Grid search over hyperparameters
for lr in 1e-3 1e-4 1e-5; do
    for hidden_size in 0 16 32 256; do
        for wd in 0.001 0.01 0.1; do
            for alpha in 0.9 1.0 1.5 2.0 3.0; do
                echo "Running with --lr $lr --hidden_size $hidden_size --wd $wd --alpha_imbalance_penalty $alpha"
                python -u /scratch/az1658/CoT_explain/20250207_R1_CoT/approximate_interm_answers/train_predictor_with_class_weights.py \
                    --epochs 200 \
                    --lr "$lr" \
                    --hidden_size "$hidden_size" \
                    --wd "$wd" \
                    --alpha_imbalance_penalty "$alpha" \
                    --threshold 0.5
            done
        done
    done
done