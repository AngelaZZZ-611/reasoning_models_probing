model_name=DeepSeek-R1-Distill-Qwen-1.5B
model_path=/path/to/your/model/$model_name
dataset=math-train
temperature=0.6
input_file=./labeled_cot/labeled_intermediate_answers_${model_name}_${dataset}_rollout_temperature${temperature}.jsonl
for file_id in {0..19}
do
python -u src/get_representation.py \
    --input_file $input_file \
    --model_name $model_path \
    --save_path ./model_embeds/${model_name}_${dataset} \
    --bs 64 \
    --file_id $file_id \
    --file_size 50 # file size measured by number of questions, should not be too large because that would affect training data shuffling
done
