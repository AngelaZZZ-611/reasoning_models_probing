model_name=DeepSeek-R1-Distill-Qwen-1.5B
model_path=/path/to/your/model/$model_name
dataset=math-train
temperature=0.6
base_name=${model_name}_${dataset}_rollout_temperature${temperature}
for chunk_id in {1..50}
do
python -u src/get_representation.py \
    --base_name $base_name \
    --model_name $model_path \
    --save_path ./model_embeds/${model_name}_${dataset} \
    --bs 64 \
    --chunk_id $chunk_id \
    --chunk_size 200 # chunk size should not be too large for training data because that would affect data shuffling
done
