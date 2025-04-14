#! /bin/bash
#SBATCH --job-name=embed
#SBATCH --nodes=1                   # Request 1 node
#SBATCH --ntasks=1                  # Run a single task
#SBATCH --gres=gpu:a100:2           # Specifically request 4 A100 GPUs
#SBATCH --cpus-per-task=16           # Number of CPU cores per task (adjust based on your needs)
#SBATCH --mem=200GB                 # Memory per node (adjust based on your needs)
#SBATCH --output=slurm-%j.log       # Standard output and error log
#SBATCH --time=24:00:00             # Time limit hrs:min:sec

model_name=DeepSeek-R1-Distill-Qwen-7B
model_path=/path/to/your/model/$model_name
dataset=math-train
temperature=0.6
base_name=${model_name}_${dataset}_rollout_temperature${temperature}
for chunk_id in {1..10}
do
python -u get_embeds.py \
    --base_name $base_name \
    --model_name $model_path \
    --save_path ./embeds_qwen1.5 \
    --bs 4 \
    --chunk_id $chunk_id \
    --chunk_size 5000 
done