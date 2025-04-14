for model in /scratch/yc7320/models/Qwen2.5-1.5B-Instruct  /scratch/yc7320/models/Qwen2.5-7B-Instruct
do
python -u generate_reasoning.py --data_name math_500 --model_path $model --temperature 0.6 
done