# Reasoning Models Know When They're Right: Probing Hidden States for Self-Verification


<div align="center">
  
[📄 Paper](https://arxiv.org/pdf/2504.05419) | [🤗 Model]() | [🤗 Data]()

</div>

Official code for ["Reasoning Models Know When They're Right: Probing Hidden States for Self-Verification"](https://arxiv.org/pdf/2504.05419) 

This repository contains code for probing hidden states of language models to predict the correctness of their intermediate answers. Our work shows that reasoning models inherently possess the capability to verify their own reasoning, with this information encoded in their hidden states.

## Table of Contents

- [Project Overview](#project-overview)
- [Setup and Requirements](#setup-and-requirements)
- [Download Trained Probes](#download-trained-probes)
- [Running the Pipeline](#running-the-pipeline)
  - [1. Generate CoT Reasoning](#1-generate-cot-reasoning) 
  - [2. Extract Reasoning Chunks](#2-extract-reasoning-chunks)
  - [3. Label Intermediate Answers](#3-label-intermediate-answers)
  - [4. Generate Hidden State Representations](#4-generate-hidden-state-representations)
  - [5. Train Probes](#5-train-probs)
  - [6. Test Probes](#6-test-probes)




## Project Overview

Our pipeline consists of several key components:
1. Generate Chain-of-Thought (CoT) reasoning paths using language models
2. Extract and segment intermediate reasoning steps
3. Label the correctness of intermediate answers using Gemini API
4. Generate hidden state representations for reasoning segments
5. Train a linear probe to predict reasoning correctness


## Setup and Requirements

```bash
# Install required packages
pip install vllm transformers torch bitsandbytes
pip install google-genai  # For labeling with Gemini API
```

## Download Trained Probes
We provide trained probes for different model and dataset combinations. You can download them using the links below:

| Model | Dataset | Download Link |
|-------|---------|---------------|
| DeepSeek-R1-Distill-Llama-8B | MATH | 🤗 Coming soon  |
| DeepSeek-R1-Distill-Llama-8B | GSM8K | 🤗 Coming soon  |
| DeepSeek-R1-Distill-Llama-8B | AIME | 🤗 Coming soon  |
| DeepSeek-R1-Distill-Llama-8B | KnowLogic | 🤗 Coming soon  |
| DeepSeek-R1-Distill-Qwen-1.5B | MATH | 🤗 Coming soon  |
| DeepSeek-R1-Distill-Qwen-1.5B | GSM8K | 🤗 Coming soon  |
| DeepSeek-R1-Distill-Qwen-1.5B | AIME | 🤗 Coming soon  |
| DeepSeek-R1-Distill-Qwen-1.5B | KnowLogic | 🤗 Coming soon  |
| DeepSeek-R1-Distill-Qwen-7B | MATH | 🤗 Coming soon  |
| DeepSeek-R1-Distill-Qwen-7B | GSM8K | 🤗 Coming soon  |
| DeepSeek-R1-Distill-Qwen-7B | AIME | 🤗 Coming soon  |
| DeepSeek-R1-Distill-Qwen-7B | KnowLogic | 🤗 Coming soon  |
| DeepSeek-R1-Distill-Qwen-32B | MATH | 🤗 Coming soon  |
| DeepSeek-R1-Distill-Qwen-32B | GSM8K | 🤗 Coming soon  |
| DeepSeek-R1-Distill-Qwen-32B | AIME | 🤗 Coming soon  |
| DeepSeek-R1-Distill-Qwen-32B | KnowLogic | 🤗 Coming soon |
| QwQ-32B | MATH | 🤗 Coming soon  |
| QwQ-32B | GSM8K | 🤗 Coming soon  |
| QwQ-32B | AIME | 🤗 Coming soon  |
| QwQ-32B | KnowLogic | 🤗 Coming soon  |


## Running the Pipeline

Below are the detailed steps to run the pipeline using DeepSeek-R1-Distill-Qwen-1.5B on the MATH dataset.

### 1. Generate CoT Reasoning
Generate Chain-of-Thought reasoning for each math problem:

```bash
# Set model path
model=/path/to/your/model/DeepSeek-R1-Distill-Qwen-1.5B

# Generate reasoning
python -u src/generate_reasoning.py \
    --data_name math-train \
    --model_path $model \
    --temperature 0.6 \
    --save_path ./initial_cot \
    --max_example 1000 # optional
```

### 2. Extract Reasoning Chunks
Process the CoT outputs to identify intermediate reasoning steps:

```bash
# Set variables
export SAVE_PATH="./processed_cot"
export NUM_PROCESSES=20  # Adjust based on CPU cores
export DATAFILE_PATH="./initial_cot/DeepSeek-R1-Distill-Qwen-1.5B_math-train_rollout_temperature0.6.jsonl"

# Process and segment reasoning paths
python src/get_reasoning_chunks.py \
    --datafile_path "$DATAFILE_PATH" \
    --save_path "$SAVE_PATH" \
    --num_processes "$NUM_PROCESSES" \
    --delete_chunks  # Optional: delete intermediate chunks after merging
```

### 3. Label Intermediate Answers
Label the correctness of intermediate answers using Gemini API:

```bash
# Set your Gemini API key
export GEMINI_API_KEY="your_api_key_here"

# Set paths
export SAVE_PATH="./labeled_cot"
export SEGMENT_PATH="./processed_cot/segmented_CoT_DeepSeek-R1-Distill-Qwen-1.5B_math-train_rollout_temperature0.6_merged.json"
export DATAFILE_PATH="./initial_cot/DeepSeek-R1-Distill-Qwen-1.5B_math-train_rollout_temperature0.6.jsonl"

# Label answers
python src/label_answer_correctness.py \
    --segmented_dataset_path "$SEGMENT_PATH" \
    --raw_CoT_path "$DATAFILE_PATH" \
    --save_path "$SAVE_PATH" \
    --num_processes 20 \
    --delete_chunks
```

### 4. Generate Hidden State Representations
Extract hidden state representations for each reasoning segment:

```bash
# Set variables
model_name=DeepSeek-R1-Distill-Qwen-1.5B
model_path=/path/to/your/model/$model_name
dataset=math-train
base_name=${model_name}_${dataset}_rollout_temperature0.6

# Generate embeddings in chunks
for chunk_id in {1..50}; do
    python -u src/get_representation.py \
        --base_name $base_name \
        --model_name $model_path \
        --save_path "./model_embeds/${model_name}_${dataset}" \
        --bs 64 \
        --chunk_id $chunk_id \
        --chunk_size 200
done
```

### 5. Train Probes
Train and evaluate the probe for predicting reasoning correctness:

```bash
# Set paths
export MODEL=DeepSeek-R1-Distill-Qwen-1.5B
export DATA=math-train
export TRAIN_DATA_PATH=./model_embeds/${MODEL}_${DATA}



# Run grid search to find optimal hyperparameters
bash grid_search.sh

# Train with best parameters (example configuration)
python -u ./src/train_predictor_with_class_weights.py \
    --epochs 200 \
    --lr 1e-5 \
    --hidden_size 0 \
    --wd 0.001 \
    --alpha_imbalance_penalty 2.0 \
    --threshold 0.5 \
    --train_data_dir $TRAIN_DATA_PATH \
    --save_model_path ./grid_search/${MODEL}_${DATA}/checkpoints \
    --store_path ./grid_search/${MODEL}_${DATA}/store \
    --model_name $MODEL
```

- For data and models used in our paper, you can replicate the results with best hyperparameters for each combination as below:

| Model | Dataset | Hidden Size | Batch Size | Learning Rate | Weight Decay | Alpha |
|-------|---------|-------------|------------|---------------|--------------|--------|
| DeepSeek-R1-Distill-Llama-8B | math-train | 0 | 64 | 1e-5 | 0.001 | 2.0 |
| DeepSeek-R1-Distill-Llama-8B | aime_83_24 | 0 | 64 | 1e-5 | 0.1 | 0.3 |
| DeepSeek-R1-Distill-Llama-8B | gsm8k-train | 16 | 64 | 1e-4 | 0.1 | 3.0 |
| DeepSeek-R1-Distill-Llama-8B | knowlogic-train | 0 | 64 | 1e-5 | 0.1 | 0.7 |
| DeepSeek-R1-Distill-Qwen-1.5B | math-train | 16 | 64 | 1e-3 | 0.01 | 2.0 |
| DeepSeek-R1-Distill-Qwen-1.5B | aime_83_24 | 16 | 64 | 1e-5 | 0.01 | 0.5 |
| DeepSeek-R1-Distill-Qwen-1.5B | gsm8k-train | 16 | 64 | 1e-5 | 0.1 | 2.0 |
| DeepSeek-R1-Distill-Qwen-1.5B | knowlogic-train | 0 | 64 | 1e-4 | 0.001 | 0.3 |
| DeepSeek-R1-Distill-Qwen-7B | math-train | 0 | 64 | 1e-4 | 0.1 | 3.0 |
| DeepSeek-R1-Distill-Qwen-7B | aime_83_24 | 0 | 64 | 1e-3 | 0.1 | 0.9 |
| DeepSeek-R1-Distill-Qwen-7B | gsm8k-train | 0 | 64 | 1e-4 | 0.1 | 3.0 |
| DeepSeek-R1-Distill-Qwen-7B | knowlogic-train | 0 | 64 | 1e-5 | 0.1 | 0.9 |
| DeepSeek-R1-Distill-Qwen-32B | math-train | 0 | 64 | 1e-4 | 0.1 | 2.0 |
| DeepSeek-R1-Distill-Qwen-32B | aime_83_24 | 0 | 64 | 1e-4 | 0.1 | 0.9 |
| DeepSeek-R1-Distill-Qwen-32B | gsm8k-train | 16 | 64 | 1e-3 | 0.001 | 3.0 |
| DeepSeek-R1-Distill-Qwen-32B | knowlogic-train | 0 | 64 | 1e-5 | 0.1 | 0.9 |
| DeepSeek-R1-Distill-Llama-70B | math-train | 0 | 64 | 1e-4 | 0.001 | 3.0 |
| DeepSeek-R1-Distill-Llama-70B | aime_83_24 | 0 | 64 | 1e-4 | 0.001 | 2.0 |
| DeepSeek-R1-Distill-Llama-70B | gsm8k-train | 0 | 64 | 1e-4 | 0.001 | 2.0 |
| DeepSeek-R1-Distill-Llama-70B | knowlogic-train | 32 | 64 | 1e-3 | 0.01 | 1.0 |

### 6. Test Probes

To test the trained probe on new data. Note that in `src/test_predictor_with_class_weights.py`, we default to use `best_val_acc` as metric for ranking the probes. You can also customize the metric.

```bash
model=DeepSeek-R1-Distill-Qwen-1.5B
data=math-train

MODEL_BASE_PATH=./grid_search/${model}_${data}
GRID_SEARCH_PATH=$MODEL_BASE_PATH/grid_search_result.jsonl
TEST_SAVE_PATH=$MODEL_BASE_PATH/test_result

TEST_DATA=./model_embeds/${model}_math_500
echo "Running test_predictor_with_class_weights.py with model $MODEL_BASE_PATH and data $TEST_DATA"
python -u ./test_predictor_with_class_weights.py \
    --input_size $INPUT_SIZE \
    --threshold 0.5 \
    --test_data_dir $TEST_DATA \
    --grid_search_result_path $GRID_SEARCH_PATH \
    --topk 10 \ # test top-k probes from grid search results
    --save_path $TEST_SAVE_PATH \
    --model_name $model
```
- To test a single probe instead of over grid search sweeps:

```bash
model=DeepSeek-R1-Distill-Qwen-1.5B
data=math-train

MODEL_BASE_PATH=./grid_search/${model}_${data}
GRID_SEARCH_PATH=$MODEL_BASE_PATH/grid_search_result.jsonl
TEST_SAVE_PATH=$MODEL_BASE_PATH/test_result

TEST_DATA=./model_embeds/${model}_math_500
echo "Running test_predictor_with_class_weights.py with model $MODEL_BASE_PATH and data $TEST_DATA"
python -u ./test_predictor_with_class_weights.py \
    --input_size $INPUT_SIZE \
    --threshold 0.5 \
    --test_data_dir $TEST_DATA \
    --save_path $TEST_SAVE_PATH \
    --model_name $model \
    --checkpoint_model_path /path/to/saved/single/pt/file 
```

## Notes

1. The pipeline is designed to process large datasets efficiently using multiprocessing
2. Intermediate results are saved after each step for reproducibility
3. For large models or long inputs, adjust batch size and chunk size based on available GPU memory
4. The Gemini API key is required for the labeling step

## Citation

If you find this code useful for your research, please cite our paper:
```bibtex
@misc{zhang2025reasoningmodelsknowtheyre,
      title={Reasoning Models Know When They're Right: Probing Hidden States for Self-Verification}, 
      author={Anqi Zhang and Yulin Chen and Jane Pan and Chen Zhao and Aurojit Panda and Jinyang Li and He He},
      year={2025},
      eprint={2504.05419},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2504.05419}, 
}
```