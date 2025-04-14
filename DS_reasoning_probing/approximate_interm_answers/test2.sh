#!/bin/bash
export CUDA_VISIBLE_DEVICES=3
# INPUT_SIZE=3584
INPUT_SIZE=1536
INPUT_SIZE=5120
INPUT_SIZE=4096
for model in DeepSeek-R1-Distill-Llama-8B 
# DeepSeek-R1-Distill-Qwen-32B
do 
for data in math-train
do
MODEL_BASE_PATH=/home/test/test05/cyl/DS_reasoning_probing/DS_reasoning_probing/grid_search/${model}_${data}
GRID_SEARCH_PATH=$MODEL_BASE_PATH/grid_search_result.jsonl
TEST_SAVE_PATH=$MODEL_BASE_PATH/test_result
# rm -rf $TEST_SAVE_PATH
# TEST_DATA=/home/test/test05/cyl/DS_reasoning_probing/DS_reasoning_probing/model_embeds/DeepSeek-R1-Distill-Qwen-7B_aime_25
for TEST_DATA in /home/test/test05/cyl/DS_reasoning_probing/DS_reasoning_probing/model_embeds/${model}_math_500 /home/test/test05/cyl/DS_reasoning_probing/DS_reasoning_probing/model_embeds/${model}_aime_25 /home/test/test05/cyl/DS_reasoning_probing/DS_reasoning_probing/model_embeds/DeepSeek-R1-Distill-Llama-8B_gsm8k-test-anqi /home/test/test05/cyl/DS_reasoning_probing/DS_reasoning_probing/model_embeds/${model}_gpqa_diamond /home/test/test05/cyl/DS_reasoning_probing/DS_reasoning_probing/model_embeds/${model}_knowlogic-test
# for TEST_DATA in /home/test/test05/cyl/DS_reasoning_probing/DS_reasoning_probing/model_embeds/DeepSeek-R1-Distill-Llama-8B_gsm8k-test-anqi
do
echo "Running test_predictor_with_class_weights.py with model $MODEL_BASE_PATH and data $TEST_DATA"
python -u ./test_predictor_with_class_weights.py \
    --input_size $INPUT_SIZE \
    --threshold 0.5 \
    --test_data_dir $TEST_DATA \
    --topk 10 \
    --save_path $TEST_SAVE_PATH \
    --model_name $model \
    --checkpoint_model_path /home/test/test05/cyl/DS_reasoning_probing/DS_reasoning_probing/grid_search/DeepSeek-R1-Distill-Llama-8B_math-train/checkpoints/best_model_weightedloss_e200-hs0-bs64-lr1e-05-wd0.001-alpha2.0-thres0.5-s42-nq1000.pt \
    --pos_weight 0.3002
done
done
done