# Description: Script to run rollout.py
model=/scratch/yc7320/models/Qwen2.5-7B-Instruct

python -u rollout.py --data_name math@train --model_path $model --temperature 1.0 --run_number 1
