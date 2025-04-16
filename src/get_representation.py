import torch
import numpy as np
import argparse
import random
import bitsandbytes as bnb
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import transformers
from torch.nn.functional import log_softmax, softmax
import json
import re
import pandas as pd
import os
from tqdm import tqdm


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)
# np.random.seed(0)
random.seed(0)


def load_model(args, model_name, bnb_config):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=True)
    n_gpus = torch.cuda.device_count()
    max_memory = "10000MB"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto", # dispatch efficiently the model on the available ressources
        max_memory = {i: max_memory for i in range(n_gpus)},
    )
    return model, tokenizer

def create_bnb_config():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    return bnb_config


def preprocess_data(args, labeled_profile):
    preprocessed_dataset = []
    sorted_keys = list(sorted(labeled_profile.keys()))
    for k in sorted_keys:
        json_object = labeled_profile[k]
        ## merge chunks
        end = -999
        last_one = len(json_object)
        for i in range(len(json_object)):
            if json_object[i]['correctness'] is not None:
                end = json_object[i]['id']
                correctness = json_object[i]['correctness']
                dict = {'id': k, 'interm_pos': end, 'correctness': correctness, 'last_one': last_one}
                preprocessed_dataset.append(dict)
    return preprocessed_dataset


def get_batch_embeddings(args, model, tokenizer, dataset, batch_size=8):
    all_last_token_embedding = []
    all_batch_info = []
    batch_prompts = []
    batch_info = []
    for item in tqdm(dataset, desc="Processing dataset"):
        for i in range(len(item["reasoning_chunks"])):
            merged_text = "\n\n".join([item["reasoning_chunks"][j] for j in range(i+1)]) + "\n\n"
            prompt = f"{item['question']}" + "\nPlease reason step by step, and put your final answer within \\boxed{}."
            input_prompt = "<｜begin▁of▁sentence｜><｜begin▁of▁sentence｜><｜User｜>" + prompt + "<｜Assistant｜><think>\n" + merged_text

            batch_prompts.append(input_prompt)
            batch_info.append(item["correctness_labels"][i])

            if len(batch_prompts) == batch_size:
                batched_last_token_embedding = run_LLM(args, batch_prompts, model, tokenizer)
                all_last_token_embedding.append(batched_last_token_embedding)
                all_batch_info.append(batch_info)
                batch_prompts = []
                batch_info = []
    if batch_prompts:
        batched_last_token_embedding = run_LLM(args, batch_prompts, model, tokenizer)
        all_last_token_embedding.append(batched_last_token_embedding)
        all_batch_info.append(batch_info)
    return all_last_token_embedding, all_batch_info


def run_LLM(args, batch_prompts, model, tokenizer):
    encoded_inputs = tokenizer(batch_prompts, return_tensors='pt', padding=True)
    input_ids = encoded_inputs.input_ids.to(device)
    # Forward pass through the model to get hidden states
    with torch.no_grad():
        # Generate output with hidden states
        outputs = model(input_ids, output_hidden_states=True)
        # Access the last hidden state
        last_hidden_state = outputs.hidden_states[-1].detach().cpu()
        # Access the logits
        # logits = outputs.logits.detach().cpu()
        del outputs  # Free GPU memory
        del input_ids
        torch.cuda.empty_cache()
    # Extract the embedding for the last token
    batched_last_token_embedding = last_hidden_state[:, -1, :]  # Batch, last token, 4096. Shape: (batch_size, sequence_length, hidden_size). For sentence embedding.
    del last_hidden_state
    # del logits
    torch.cuda.empty_cache()
    return batched_last_token_embedding



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='DeepSeek-R1-Distilled-8B',
                        help='LLM model used: DeepSeek-R1-Distilled-8B, ..')
    parser.add_argument('--input_file', type=str, default='./labeled_cot/labeled_intermediate_answers_DeepSeek-R1-Distill-Qwen-1.5B_math-train_rollout_temperature0.6.jsonl')


    parser.add_argument('--save_path', type=str, default='./model_embeds', help='save path for the embeddings. ')
    parser.add_argument('--bs', type=int, default=2,
                        help='batch size to get the embeddings, e.g., 2/4/8, depends on gpu types. ')
    parser.add_argument('--file_id', type=int, default=0)
    parser.add_argument('--file_size', type=int, default=0)
    args = parser.parse_args()

    os.makedirs(args.save_path, exist_ok=True)
    ########## load data files
    dataset = [json.loads(line) for line in open(args.input_file).readlines()]

    dataset = dataset[args.file_id*args.file_size: (args.file_id+1)*args.file_size]
    
    if not dataset:
        import sys
        sys.exit(0)

    # model_name = args.cache_dir
    bnb_config = create_bnb_config()
    model, tokenizer = load_model(args, args.model_name, bnb_config)

    ### run LLM: get intermediate answer embeddings
    all_last_token_embedding, all_batch_info = get_batch_embeddings(args, model, tokenizer, dataset, batch_size=args.bs)

    ### store
    all_last_token_embedding = torch.cat(all_last_token_embedding, dim=0) # save space
    profile = {'all_last_token_embedding': all_last_token_embedding, 'all_batch_info': all_batch_info}
    print(f"number of embeddings: {all_last_token_embedding.shape[0]}")
    print(f"saved to {args.save_path}/embed_file_{args.file_id*args.file_size}_{(args.file_id+1)*args.file_size}.pt")
    torch.save(profile, f'{args.save_path}/embed_file_{args.file_id*args.file_size}_{(args.file_id+1)*args.file_size}.pt')


if __name__ == '__main__':
    main()