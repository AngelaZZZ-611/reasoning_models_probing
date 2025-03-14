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
    not_finished_examples = []
    for k in labeled_profile.keys():
        output = labeled_profile[k].text
        # convert the JSON formatted text string into a JSON object
        try:
            json_string = output.replace('```json\n', '').replace('\n```', '')
            json_string = json_string.replace('None', 'null')
            json_string = json_string.replace('False', 'false')
            json_string = json_string.replace('True', 'true')
            json_object = json.loads(json_string)
            # preprocessed_dataset[k] = []
        except:
            not_finished_examples.append(k)
            continue  # for these examples, record the idx, and later run-label by using a longer gen-length
        ## merge chunks
        init = 1
        end = -999
        last_one = len(json_object)
        for i in range(len(json_object)):
            if json_object[i]['correctness'] != None:
                end = json_object[i]['id']
                correctness = json_object[i]['correctness']
                # dict = {'id': k, 'range': [str(init), end], 'correctness': correctness, 'last_one': last_one}
                dict = {'id': k, 'interm_pos': end, 'correctness': correctness, 'last_one': last_one}
                preprocessed_dataset.append(dict)
                # init = int(end) + 1
            # if i == len(json_object): # if finally no results (not finish generation) --> take it as False (weird)
            #     dict = {'id': k, 'interm_pos': last_one, 'correctness': False, 'last_one': last_one}
            #     preprocessed_dataset.append(dict)
    print(not_finished_examples)
    # print(len(preprocessed_dataset)) 
    # store = {'not_finished_examples': not_finished_examples, 'preprocessed_dataset': preprocessed_dataset}
    # torch.save(store, f'{args.dataset_path}/label_intermediate_answers/train_dataset_MATH/store_preprocessed_seg_ranges_CoT_MATH-7474_10000_temp0.6_{100*args.st}.pt')
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
            # id_range = batch_collector[j]['range']
            interm_pos = batch_collector[j]['interm_pos']
            correctness = batch_collector[j]['correctness']
            # Generate a list of consecutive integers
            keys_to_merge = [str(i) for i in range(1, int(interm_pos) + 1)]
            # merge steps into a string with \n\n
            merged_text = "\n\n".join([data[data_id][key] for key in keys_to_merge]) +"\n\n" 
            # recover previous full-formatted text (i.e., keep same with the one in generation)
            prompt = f"{orig_data[int(data_id)]['problem']}" + "\n\nPlease reason step by step, and put your final answer within \\boxed{}."
            # messages = [{"role": "user", "content": prompt},
            #             {"role": "assistant", "content": merged_text}]
            # input_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            # just maunally make sure this is exactly same format with the one during raw CoT generation
            input_prompt = "<｜begin▁of▁sentence｜><｜begin▁of▁sentence｜><｜User｜>" + prompt + "<｜Assistant｜><think>\n" + merged_text
            batch_prompts.append(input_prompt)
        
        # run LLM
        batched_last_token_embedding = run_LLM(args, batch_prompts, model, tokenizer)

        all_last_token_embedding.append(batched_last_token_embedding)
        all_batch_info.append(batch_collector)
    return all_last_token_embedding, all_batch_info


def run_LLM(args, batch_prompts, model, tokenizer):
    encoded_inputs = tokenizer(batch_prompts, return_tensors='pt') # Long input text
    input_ids = encoded_inputs.input_ids.to(device)

    chunk_size = 512 # Adjust based on GPU memory
    chunks = [input_ids[:, i:i+chunk_size] for i in range(0, input_ids.shape[1], chunk_size)]
    total_chunks = len(chunks)

    past_key_values = None
    last_hidden_state = None

    for i, chunk in enumerate(chunks):
        # Calculate position_ids for the current chunk
        start_pos = i * chunk_size
        position_ids = torch.arange(start_pos, start_pos + chunk.size(1), device=model.device).unsqueeze(0)

        if i == (total_chunks-1):
            # Forward pass through the model to get hidden states
            with torch.no_grad():
                # Generate output with hidden states
                outputs = model(
                    input_ids=chunk,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    use_cache=True,  # Retain KV cache for next chunks
                    output_hidden_states=True
                )
                last_hidden_state = outputs.hidden_states[-1].detach().cpu()
                del outputs  # Free GPU memory
                del chunk
                torch.cuda.empty_cache()
        else:
            # Forward pass through the model to get hidden states
            with torch.no_grad():
                # Generate output with hidden states
                outputs = model(
                    input_ids=chunk,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    use_cache=True  # Retain KV cache for next chunks
                )
                past_key_values = outputs.past_key_values
                del outputs  # Free GPU memory
                del chunk
                torch.cuda.empty_cache()

    del input_ids
    # Extract the embedding for the last token
    batched_last_token_embedding = last_hidden_state[:, -1, :]  # Batch, last token, 4096. Shape: (batch_size, sequence_length, hidden_size). For sentence embedding.
    del last_hidden_state
    torch.cuda.empty_cache()
    return batched_last_token_embedding



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='DeepSeek-R1-Distilled-8B',
                        help='LLM model used: DeepSeek-R1-Distilled-8B, ..')
    parser.add_argument('--llm_name', type=str, default='deepseek-r1-8B',
                        help='LLM model used.')
    parser.add_argument('--cache_dir', type=str, default='/scratch/az1658/LLMs/DeepSeek/models--deepseek-ai--DeepSeek-R1-Distill-Llama-8B/snapshots/81cee02dd020268dced5fa1327e8555acce9c63c', help='model cache dir.')
    parser.add_argument('--dataset_path', type=str, default='/scratch/az1658/CoT_explain/20250207_R1_CoT/20250307_OOD/profile_CoT_generation', help='input dataset path.') 
    parser.add_argument('--save_path', type=str, default='/scratch/az1658/CoT_explain/20250207_R1_CoT/20250307_OOD/profile_CoT_generation/embeds_intermediate_answers', help='save path for the embeddings. ')
    parser.add_argument('--data_name', type=str, default='AIME', help='datasets, e.g., MATH, AIME, GSM8k-test..')
    parser.add_argument('--data_size', type=int, default=30, help='total number of data in this dataset, e.g., MATH:500, AIME:30, GSM8k-test:1319..')
    parser.add_argument('--temperature', type=float, default=0.7, 
                        help='temperature setup for generation, the larger the diversity. ')
    # parser.add_argument('--top_p', type=float, default=0.99,
    #                     help='top_p setup for generation, the larger the diversity. ')
    parser.add_argument('--max_new_tokens', type=int, default=40000,
                        help='max_new_tokens setup for model.generate().')
    parser.add_argument('--st', type=int, default=0,
                        help='start idx.')
    parser.add_argument('--bs', type=int, default=2,
                        help='batch size to get the embeddings, e.g., 2/4/8, depends on gpu types. ')
    args = parser.parse_args()

    model_name = args.cache_dir
    bnb_config = create_bnb_config()
    model, tokenizer = load_model(args, model_name, bnb_config)

    ########## load data files: segmented, labeled-intermeidate-answers, raw-CoT files
    ed = min(100*args.st+100, args.data_size)
    print(f'==running {100*args.st} to {ed}==')
    # load sengemented CoT files
    file_path = f'{args.dataset_path}/segmented/{args.data_name}/segmented_CoT_{args.data_name}_40000_temp0.7_{100*args.st}_{ed}.json' 
    with open(file_path, 'r') as f:
        data = json.load(f)
    # load the labeled intermediate answer files
    labeled_profile = torch.load(f'{args.dataset_path}/label_intermediate_answers/{args.data_name}/labeled_intermidiate_answers_CoT_{args.data_name}_40000_temp0.7_{100*args.st}_{ed}.pt')
    # load raw CoT geenration data for questions
    orig_data = torch.load(f'{args.dataset_path}/raw_CoT/{args.data_name}/CoT_generation_{args.data_name}_40000_temp0.7_{100*args.st}_{ed}.pt')

    ### preprocess and merge chunks
    preprocessed_dataset = preprocess_data(args, labeled_profile)

    ### run LLM: get intermediate answer embeddings
    all_last_token_embedding, all_batch_info = get_batch_embeddings(args, model, tokenizer, data, orig_data, preprocessed_dataset, batch_size=args.bs)

    ### store
    all_last_token_embedding = torch.cat(all_last_token_embedding, dim=0) # save space
    profile = {'all_last_token_embedding': all_last_token_embedding, 'all_batch_info': all_batch_info}

    torch.save(profile, f'{args.save_path}/{args.data_name}/embeds-CoT_generation_{args.data_name}_40000_temp{args.temperature}_{100*args.st}_{ed}.pt')


if __name__ == '__main__':
    main()