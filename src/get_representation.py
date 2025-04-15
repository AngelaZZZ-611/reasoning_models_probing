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


def get_batch_embeddings(args, model, tokenizer, data, orig_data, preprocessed_dataset, batch_size=8):
    all_last_token_embedding = []
    all_batch_info = []
    for i in range(0, len(preprocessed_dataset), batch_size):
        batch_collector = preprocessed_dataset[i:i+batch_size]
        print(f'==i: {i}')
        print(batch_collector)

        # format the prompts together with question before the gen-CoT
        batch_prompts = []
        for j in range(len(batch_collector)):
            data_id = batch_collector[j]['id']
            interm_pos = batch_collector[j]['interm_pos']
            correctness = batch_collector[j]['correctness']
            # Generate a list of consecutive integers
            keys_to_merge = [str(i) for i in range(1, int(interm_pos) + 1)]
            # merge steps into a string with \n\n
            merged_text = "\n\n".join([data[data_id][key] for key in keys_to_merge]) +"\n\n" 
            # recover previous full-formatted text (i.e., keep same with the one in generation)
            # TODO: flexible template
            prompt = f"{orig_data[str(data_id)]['instruction']}" + "\nPlease reason step by step, and put your final answer within \\boxed{}."

            # just maunally make sure this is exactly same format with the one during raw CoT generation
            input_prompt = "<｜begin▁of▁sentence｜><｜begin▁of▁sentence｜><｜User｜>" + prompt + "<｜Assistant｜><think>\n" + merged_text
            batch_prompts.append(input_prompt)
        
        # run LLM
        batched_last_token_embedding = run_LLM(args, batch_prompts, model, tokenizer)

        all_last_token_embedding.append(batched_last_token_embedding)
        all_batch_info.append(batch_collector)
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
    parser.add_argument('--base_name', type=str, default='DeepSeek-R1-Distill-Qwen-7B_math-train_rollout_temperature0.6',
                        help='LLM model used.')

    parser.add_argument('--save_path', type=str, default='./embeds_intermediate_answers', help='save path for the embeddings. ')
    parser.add_argument('--bs', type=int, default=2,
                        help='batch size to get the embeddings, e.g., 2/4/8, depends on gpu types. ')
    parser.add_argument('--chunk_id', type=int, default=0,
                        help='batch size to get the embeddings, e.g., 2/4/8, depends on gpu types. ')
    parser.add_argument('--chunk_size', type=int, default=0,
                        help='batch size to get the embeddings, e.g., 2/4/8, depends on gpu types. ')
    args = parser.parse_args()

    os.makedirs(args.save_path, exist_ok=True)

    # model_name = args.cache_dir
    bnb_config = create_bnb_config()
    model, tokenizer = load_model(args, args.model_name, bnb_config)

    ########## load data files: segmented, labeled-intermeidate-answers, raw-CoT files
    file_path = f"./processed_cot/segmented_CoT_{args.base_name}_merged.json"

    with open(file_path, 'r') as f:
        data = json.load(f)
    data = {str(k): v for k, v in data.items()} 

    # load the labeled intermediate answer files
    labeled_profile = torch.load(f'./labeled_cot/labeled_intermidiate_answers_{args.base_name}_merged.pt')
    # load raw CoT geenration data for questions
    orig_data = [json.loads(line) for line in open(f'./initial_cot/{args.base_name}.jsonl').readlines()]
    orig_data = {str(item['id']): item for item in orig_data}

    ### preprocess and merge chunks
    preprocessed_dataset = preprocess_data(args, labeled_profile)

    preprocessed_dataset = sorted(preprocessed_dataset, key=lambda x: (x['id'], x["interm_pos"]))
    print("total number of preprocessed data: ", len(preprocessed_dataset))

    preprocessed_dataset = preprocessed_dataset[args.chunk_id*args.chunk_size: (args.chunk_id+1)*args.chunk_size]
    print("current chunk of preprocessed data: ", len(preprocessed_dataset))
    if not preprocessed_dataset:
        import sys
        sys.exit(0)

    ### run LLM: get intermediate answer embeddings
    all_last_token_embedding, all_batch_info = get_batch_embeddings(args, model, tokenizer, data, orig_data, preprocessed_dataset, batch_size=args.bs)

    ### store
    all_last_token_embedding = torch.cat(all_last_token_embedding, dim=0) # save space
    profile = {'all_last_token_embedding': all_last_token_embedding, 'all_batch_info': all_batch_info}

    torch.save(profile, f'{args.save_path}/{args.base_name}_chunk_{args.chunk_id*args.chunk_size}_{(args.chunk_id+1)*args.chunk_size}.pt')


if __name__ == '__main__':
    main()