#!/bin/bash
export CUDA_VISIBLE_DEVICES=6

REPO_DIR=/home/test/test05/cyl/DS_reasoning_probing/DS_reasoning_probing
OFFPOLICY_EMBEDS_DIR=$REPO_DIR/model_embeds_offpolicy

MODEL=DeepSeek-R1-Distill-Qwen-1.5B


for offpolicy_model in DeepSeek-R1-Distill-Qwen-32B DeepSeek-R1-Distill-Qwen-7B DeepSeek-R1-Distill-Llama-8B DeepSeek-R1-Distill-Llama-70B QwQ-32B
# for offpolicy_model in DeepSeek-R1-Distill-Llama-8B DeepSeek-R1-Distill-Llama-70B
do 
for train_data in math-train_by_batch
do

MODEL_BASE_PATH=/home/test/test05/cyl/DS_reasoning_probing/DS_reasoning_probing/grid_search/${MODEL}_${train_data}
GRID_SEARCH_PATH=$MODEL_BASE_PATH/grid_search_result.jsonl

TEST_SAVE_PATH=$REPO_DIR/offpolicy_test_result

# rm -rf $TEST_SAVE_PATH
for TEST_DATA in ${OFFPOLICY_EMBEDS_DIR}/${offpolicy_model}_aime_25_embedby_${MODEL}

do
echo "Running test_predictor_with_class_weights.py with model $MODEL_BASE_PATH and data $TEST_DATA"
python -u ./test_predictor_with_class_weights.py \
    --threshold 0.5 \
    --test_data_dir $TEST_DATA \
    --grid_search_result_path $GRID_SEARCH_PATH \
    --topk 10 \
    --save_path $TEST_SAVE_PATH \
    --model_name $MODEL
done
done
done