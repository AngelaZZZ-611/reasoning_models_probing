import torch
import numpy as np
import argparse
import random
import bitsandbytes as bnb
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import transformers
from torch.nn.functional import log_softmax, softmax
import json
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


def load_dataset(dataset_path):    
    loaded_dataset = {}
    df = pd.read_parquet(dataset_path)
    for i in range(len(df)):
        each_data = {}
        each_data['problem'] = df.iloc[i]['problem']
        each_data['answer'] = df.iloc[i]['answer']
        each_data['subject'] = df.iloc[i]['subject']
        each_data['unique_id'] = df.iloc[i]['unique_id']
        loaded_dataset[i] = each_data
    return loaded_dataset

def load_train_dataset(dataset_path):    
    loaded_dataset = {}
    with open(dataset_path, 'r') as f:
        data = json.load(f)
    for i in range(len(data)):
        each_data = {}
        each_data['problem'] = data[i]['problem']
        each_data['answer'] = data[i]['answer']
        each_data['subject'] = data[i]['type']
        # each_data['unique_id'] = df.iloc[i]['unique_id']
        loaded_dataset[i] = each_data
    return loaded_dataset

def run_LLM(args, data, model, tokenizer):
    prompt = f"{data['problem']}" + "\n\nPlease reason step by step, and put your final answer within \\boxed{}."
    # prompt = f"{data['problem']}" + "\n\nPlease reason step by step, and put your final answer within \\boxed{}. Moreover, during the <think> process, put any of your intermediate answers/results before reflection/verification within **{}**."  # useless
    # prompt = f"{data['problem']}" + "\n\nPlease reason step by step, and put your intermediate answers and final answer within \\boxed{}." # useless

    messages = [{"role": "user", "content": prompt}]
    input_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    encoded_inputs = tokenizer(input_prompt, return_tensors='pt')
    input_ids = encoded_inputs.input_ids.to(device)
    if args.temperature == 0.0:
        outputs = model.generate(input_ids, do_sample=False, max_new_tokens=args.max_new_tokens)
    else:
        outputs = model.generate(input_ids, do_sample=True, temperature=args.temperature, max_new_tokens=args.max_new_tokens)
    del input_ids
    torch.cuda.empty_cache()
    res = tokenizer.decode(outputs[0])
    del outputs
    torch.cuda.empty_cache()
    generated_sentence = res.split('<｜Assistant｜>')[1]
    if len(generated_sentence.split('</think>'))>1:
        answer = generated_sentence.split('</think>')[1].strip()
    else:
        answer = None
    return generated_sentence, answer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='DeepSeek-R1-Distilled-8B',
                        help='LLM model used: DeepSeek-R1-Distilled-8B, ..')
    parser.add_argument('--llm_name', type=str, default='deepseek-r1-8B',
                        help='LLM model used.')
    parser.add_argument('--cache_dir', type=str, default='/scratch/az1658/LLMs/DeepSeek/models--deepseek-ai--DeepSeek-R1-Distill-Llama-8B/snapshots/81cee02dd020268dced5fa1327e8555acce9c63c', help='model cache dir.')
    # parser.add_argument('--dataset_path', type=str, default='/scratch/az1658/CoT_explain/datasets/MATH-500/0000.parquet', help='input dataset path.') # testset
    parser.add_argument('--dataset_path', type=str, default='/scratch/az1658/CoT_explain/datasets/MATH-500/math.json', help='input dataset path.') # train-set
    parser.add_argument('--save_path', type=str, default='/scratch/az1658/CoT_explain/20250207_R1_CoT/profile_CoT_generation/raw_CoT', help='save path for the CoT generation. ')
    parser.add_argument('--temperature', type=float, default=0.6, 
                        help='temperature setup for generation, the larger the diversity. ')
    # parser.add_argument('--top_p', type=float, default=0.99,
    #                     help='top_p setup for generation, the larger the diversity. ')
    parser.add_argument('--max_new_tokens', type=int, default=10000,
                        help='max_new_tokens setup for model.generate().')
    parser.add_argument('--st', type=int, default=0,
                        help='start idx.')
    args = parser.parse_args()

    model_name = args.cache_dir
    bnb_config = create_bnb_config()
    model, tokenizer = load_model(args, model_name, bnb_config)

    # load the dataset
    if args.dataset_path.endswith('.parquet'): # MATH-500 (testset)
        preprocessed_dataset = load_dataset(args.dataset_path)
    elif args.dataset_path.endswith('.json'): # MATH (trainset)
        preprocessed_dataset = load_train_dataset(args.dataset_path)

    # split entire dataset into job-arrays to run and save
    profile={'args':args}
    total_number = len(preprocessed_dataset)
    print(total_number)
    uid_list = list(preprocessed_dataset.keys())
    # ed = total_number
    # current_running = uid_list[args.st: ed] 
    ed = min(100*args.st+100, total_number) # split job-array
    current_running = uid_list[100*args.st: ed] 
    print(f'==running {args.st} to {ed}==')

    # inference to get the generated CoTs
    for uid in current_running:
        print(f'==uid: {uid}')
        profile[uid] = {}
        data = preprocessed_dataset[uid]
        # run LLM (for context generation results)
        generated_sentence, answer = run_LLM(args, data, model, tokenizer)
        profile[uid]['problem'] = data['problem']
        profile[uid]['gt_answer'] = data['answer']
        # profile[uid]['unique_id'] = data['unique_id']
        profile[uid]['generated_sentence'] = generated_sentence
        profile[uid]['answer'] = answer

    # save
    if args.dataset_path.endswith('.parquet'): # MATH-500 (testset)
        torch.save(profile, f'{args.save_path}/test_dataset_MATH/CoT_generation_MATH-500_{args.max_new_tokens}_temp{args.temperature}_{100*args.st}_{ed}.pt')
    elif args.dataset_path.endswith('.json'): # MATH (trainset)
        torch.save(profile, f'{args.save_path}/train_dataset_MATH/CoT_generation_MATH-7474_{args.max_new_tokens}_temp{args.temperature}_{100*args.st}_{ed}.pt')



if __name__ == '__main__':
    main()