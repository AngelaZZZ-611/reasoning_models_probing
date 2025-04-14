#!/bin/bash

# Configuration - modify these variables as needed
SAVE_PATH="./processed_cot_new"
NUM_PROCESSES=20  # Use number of CPU cores
DELETE_CHUNKS=true      # Set to false if you want to keep individual chunk files

# Create the save directory if it doesn't exist
mkdir -p "$SAVE_PATH"



# Set up delete_chunks flag
DELETE_FLAG=""
if [ "$DELETE_CHUNKS" = true ]; then
    DELETE_FLAG="--delete_chunks"
fi
LLAMA_MODELS=("DeepSeek-R1-Distill-Llama-8B")
# LLAMA_MODELS=("DeepSeek-R1-Distill-Llama-8B")

QWEN_MODELS=("DeepSeek-R1-Distill-Qwen-1.5B" "DeepSeek-R1-Distill-Qwen-7B" "DeepSeek-R1-Distill-Qwen-32B")

for model in QwQ-32B
do
for data in gpqa_diamond
do
DATAFILE_PATH="./initial_cot/${model}_${data}_rollout_temperature0.6.jsonl"
# Print execution information
echo "Starting processing with the following configuration:"
echo "Data file path: $DATAFILE_PATH"
echo "Save path: $SAVE_PATH"
echo "Number of processes: $NUM_PROCESSES"
echo "Delete chunks after merging: $DELETE_CHUNKS"
# Run the Python script
echo "Running data processing..."
python find_intermidiate_answers.py \
    --datafile_path "$DATAFILE_PATH" \
    --save_path "$SAVE_PATH" \
    --num_processes "$NUM_PROCESSES" \
    $DELETE_FLAG

echo "All tasks complete"
done
done