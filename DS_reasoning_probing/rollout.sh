for model in /path/to/model/Qwen2.5-1.5B-Instruct  /path/to/model/Qwen2.5-7B-Instruct
do
python -u generate_reasoning.py --data_name math_500 --model_path $model --temperature 0.6 --save_path initial_cot
done