
model=/path/to/your/model/DeepSeek-R1-Distill-Qwen-1.5B
python -u src/generate_reasoning.py --data_name math@train --model_path $model --temperature 0.6 --save_path ./initial_cot --max_example 1000