# Experiment process overview

1. Run each question on model to get the CoT generated

2. Find out the intermediate answer positions within each CoT, reorganize (i.e., including split and merge) the CoT into several reasoning paths, and label its correctness according to rule-based measurement (i.e., True|False)

3. Get the embeddings of input chunks, then
    Craft the dataset by
    X: last-token’s last-hidden-states of each input-text chunk (i.e., embedding); 
    Y: True | False (e.g., the correctness of the intermediate answer at the end of each chunk)

4. Train a predictor, and test to see its performance on predicting the correctness of intermediate/final answers at the answer positions.

## A word of caution (write before): 

1. In order to run with more nodes, I split the entire dataset into several job-arrays and submit them together. Of course, you can modify this part according to your own habits (or according to the total amount of other datasets).  -- this part (about separate data to run) should be easy to find in the code.

2. I saved the results for every middle-phase (you will know what I mean when looking down). So when running any follow-up phase, don't forget to parser-in the path for previous saved files. (Of course, if you are used to putting them together/ignore middle-phase profiles/.., feel free to modify it by yourself)


## S1. Run each question on model to get the CoT generated

code: CoT_generation.py

```bash
#SBATCH --array=0-4
python -u .../CoT_generation.py --st ${SLURM_ARRAY_TASK_ID} --cache_dir "note:model path" --dataset_path "note:data path" --save_path "note:save-file"
```
> for running faster, I split the data into job-arrays (but also depends on how many gpus are avaiable to use). Anyway, feel free to change this part when you run experiments.

See `rollout.sh` for faster inference with vllm library. Install the lib before run.
- Note: probably need to adjust max_new_token and other parameter to cater for reasoning models.
```bash
pip install vllm
```

## S2. Split reasoning paths and label them

code: find_intermidiate_answers.py -> label_intermidiate_answers_api.py

```bash
#SBATCH --gres=gpu:0 # just cpu
#SBATCH --array=0-4
python -u .../find_intermidiate_answers.py --st ${SLURM_ARRAY_TASK_ID} --datafile_path "note:raw CoT path" --save_path "note:save-file"
```
> First, find out the intermediate answer positions (reasoning paths) within each CoT by keywords pattern matching. "keywords" setted here usually lead to another reasoning path. 

```bash
#SBATCH --gres=gpu:0 # just cpu
#SBATCH --array=0-4
python -u .../label_intermidiate_answers_api.py --st ${SLURM_ARRAY_TASK_ID} --segmented_dataset_path "note:segmented CoT file path" --raw_CoT_path "note:raw CoT path" --save_path "note:save-file"
```
> Then, use LLM (Gemini API) to help extract the intermediate/final answers, as well as compared with ground-truth answer. 


## S3. Get the embedding vector of each input reasoning path.

code: get_inter_ans_embeds.py  (get_inter_ans_embeds-kvcache.py)

```bash
#SBATCH --array=0-4
python -u .../get_inter_ans_embeds.py --st ${SLURM_ARRAY_TASK_ID} --bs "note:batch-size" --cache_dir "note:model path" --dataset_path "note:data path" --save_path "note:save-file"
```
> Merge chunks; Get the embedding of each input reasoning path; Labels are also contained in the final saved file.

```bash
python .../get_inter_ans_embeds-kvcache.py ...
```
> obtain the embedding of the last token for very long inputs without running into OOM errors, i.e., Chunked Processing with Key-Value Caching.

## S4. Train the predictor

code (in directory -- approximate_interm_answers): train_predictor_with_class_weights.py ; test_predictor_with_class_weights.py ; dataloader.py

```bash
bash .../approximate_interm_answers/grid_search.sh
```
> First, train the predictor, and use grid-search to decide the model architecture and hyperparameters based on the performance of validation data.

```bash
python -u .../approximate_interm_answers/train_predictor_with_class_weights.py --epochs 200 --lr 1e-5 --wd 0.001 --hidden_size 0 --alpha_imbalance_penalty 2.0 
```
> P.s., this is the setup finally chosen for MATH dataset.

```bash
python test_predictor_with_class_weights.py --lr 1e-05 --wd 0.001 --hidden_size 0 --alpha_imbalance_penalty 2.0 --checkpoint_model_path "node:best-model path"
```
> run test only on checkpoint models (e.g., the best model)

