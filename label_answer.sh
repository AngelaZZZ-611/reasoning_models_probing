export GEMINI_API_KEY="your_gemini_api_key"

SAVE_PATH="./labeled_cot"
NUM_PROCESSES=20  # Use number of CPU cores
DELETE_CHUNKS=true     # Set to false if you want to keep individual chunk files

# Create the save directory if it doesn't exist
mkdir -p "$SAVE_PATH"



# Set up delete_chunks flag
DELETE_FLAG=""
if [ "$DELETE_CHUNKS" = true ]; then
    DELETE_FLAG="--delete_chunks"
fi
# gsm8k-train gsm8k-test aime_83_24 aime_25 math-train 
# LLAMA_MODELS=("DeepSeek-R1-Distill-Llama-8B" "DeepSeek-R1-Distill-Llama-70B")
LLAMA_MODELS=("DeepSeek-R1-Distill-Llama-8B")

QWEN_MODELS=("DeepSeek-R1-Distill-Qwen-1.5B" "DeepSeek-R1-Distill-Qwen-7B" "DeepSeek-R1-Distill-Qwen-32B")

# for model in "${LLAMA_MODELS[@]}" "${QWEN_MODELS[@]}"
for model in DeepSeek-R1-Distill-Llama-70B
do
for data in knowlogic-train knowlogic-test 
do
DATAFILE_PATH="./initial_cot/${model}_${data}_rollout_temperature0.6.jsonl"
SEGMENT_PATH="./processed_cot/segmented_CoT_${model}_${data}_rollout_temperature0.6_merged.json"
# Print execution information
echo "Starting processing with the following configuration:"
echo "Data file path: $DATAFILE_PATH"
echo "Save path: $SAVE_PATH"
echo "Number of processes: $NUM_PROCESSES"
echo "Delete chunks after merging: $DELETE_CHUNKS"
# Run the Python script
echo "Running data processing..."
python src/label_answer_correctness.py \
    --segmented_dataset_path "$SEGMENT_PATH" \
    --raw_CoT_path "$DATAFILE_PATH" \
    --save_path "$SAVE_PATH" \
    --num_processes "$NUM_PROCESSES" \
    $DELETE_FLAG

echo "All tasks complete"
done
done