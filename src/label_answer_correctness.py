import torch
import numpy as np
import argparse
import random
import transformers
import json
import re
import pandas as pd
import os

from google import genai
from google.genai import types
from tqdm import tqdm
import json
import glob
from multiprocessing import Pool, cpu_count
from utils import process_json_output
# Set your Gemini API key as an environment variable 
# os.environ["GEMINI_API_KEY"] = "" #"your_api_key_here" 



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)
# np.random.seed(0)
random.seed(0)


INSTRUCT_PROMPT = """Given several chunks of a reasoning trace, as well as a ground-truth answer. Independently evaluate each chunk, if each chunk reaches to a result at the end of this chunk, return the intermediate result; otherwise return None, if the chunk does not contain an intermediate result(e.g., pure reflections).

Then, if the intermediate answer exists, compare it to the ground-truth answer. If the intermediate result in the chunk equals to the ground-truth answer, return True; If the intermeidate result in the chunk does not euqal to the ground-truth answer, return False; If no intermediate answer, return None.

Output in a JSON format:
[  
  {"id": "1", "result": "6 + 9i"/None, "correctness": True/False/None},
  ...  
] 
"""

def run_LLM(args, client, reasoning_trace, gt_answer):
    # create prompt
    prompt = INSTRUCT_PROMPT + f"Input chunks: {reasoning_trace}" + f"\n\nGround-truth answer: {gt_answer}"
    # send request
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=[prompt],
        config=types.GenerateContentConfig(
            max_output_tokens=10000, #1000 is not enough for very long reasoning traces, maybe 10000
            temperature=args.temperature
        )
    )

    return response



# response.text = '```json\n[\n  {"id": "1", "result": "5/3", "correctness": null},\n  {"id": "2", "result": "14/3", "correctness": true},\n  {"id": "3", "result": "14/3", "correctness": true},\n  {"id": "4", "result": null, "correctness": null},\n  {"id": "5", "result": null, "correctness": null},\n  {"id": "6", "result": null, "correctness": null},\n  {"id": "7", "result": "14/3", "correctness": true},\n  {"id": "8", "result": null, "correctness": null},\n  {"id": "9", "result": "\\\\dfrac{14}{3}", "correctness": true}\n]\n```'

def load_id2data(file_path):
    id2data = {}
    with open(file_path, 'r') as f:
        for line in f:
            item = json.loads(line)
            id2data[str(item['id'])] = item
    return id2data


def load_id2answer(file_path):
    id2ans = {}
    with open(file_path, 'r') as f:
        for line in f:
            item = json.loads(line)
            if "gpqa" in file_path:
                id2ans[str(item['id'])] = f"({item['answer']}) " + item["Correct Answer"]
            elif "knowlogic" in file_path:
                id2ans[str(item['id'])] = f"({item['answer']}) " + item["options"][item['answer']]
            else:
                id2ans[str(item['id'])] = item['answer']
    return id2ans

def split_data_for_parallel(data_file, num_chunks):
    """Split the data file into chunks for parallel processing."""
    keys = list(sorted(data_file.keys()))
    chunk_size = max(1, len(keys) // num_chunks + 1)
    
    chunks = []
    for i in range(0, len(keys), chunk_size):
        chunk_keys = keys[i:i+chunk_size]
        chunk_data = {k: data_file[k] for k in chunk_keys if k in data_file}
        chunks.append(chunk_data)
    
    return chunks

def valid_label(res, length):
    if res is None:
        return False
    for label in res:
        if int(label["id"]) > length or int(label["id"]) < 1:
            return False
    return True
    

def single_process(args_dict):
    """Process a single chunk of data."""
    args = args_dict['args']
    chunk_id = args_dict['chunk_id']
    id2ans = args_dict['id2ans']
    id2reasoning = args_dict['id2reasoning']
    api_key = os.getenv("GEMINI_API_KEY")
    client = genai.Client(api_key=api_key)
    # process prompt and run LLM
    results = {}
    for k in tqdm(id2reasoning.keys()):
        reasoning_trace = id2reasoning[k]
        gt_answer = id2ans[str(k)]
        out_res = run_LLM(args, client, reasoning_trace, gt_answer)
        out_res = process_json_output(out_res.text)
        if not valid_label(out_res, len(reasoning_trace)):
            print(f"Error processing output for {k}")
            continue
        results[k] = out_res
    
    return results

def merge_reasoning_chunks(reasoning_chunks, labels):
    # reasoning_chunks:  list[str]
    # labels: list[dict]
    # return: list[str], list[dict]
    # merge reasoning chunks to make sure each new chunk has a label[correctness] that is not None
    all_merged_reasoning_chunks = []
    all_merged_labels = []
    labels = sorted(labels, key=lambda x: int(x["id"]))
    prev_idx = 1  # start from 1
    for label in labels:
        if label["correctness"] is not None:
            new_idx = int(label["id"]) + 1
            current_chunk = []
            for i in range(prev_idx, new_idx):
                current_chunk.append(reasoning_chunks[str(i)])

            all_merged_reasoning_chunks.append("\n\n".join(current_chunk))
            label.pop("id")
            all_merged_labels.append(label)
            prev_idx = new_idx
    assert len(all_merged_reasoning_chunks) == len(all_merged_labels), f"Length of reasoning chunks and labels do not match, reasoning_chunks: {len(all_merged_reasoning_chunks)}, labels: {len(all_merged_labels)}"
    return all_merged_reasoning_chunks, all_merged_labels



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--segmented_dataset_path', type=str, default=None, help='segmented dataset path.')
    parser.add_argument('--raw_CoT_path', type=str, default=None, help='raw CoT dataset path.')
    parser.add_argument('--save_path', type=str, default=None, help='save path.')
    parser.add_argument('--temperature', type=float, default=0.6, 
                        help='temperature setup for generation, the larger the diversity. ')
    parser.add_argument('--num_processes', type=int, default=None, help='Number of processes to use. Defaults to number of CPU cores.')
    parser.add_argument('--num_chunks', type=int, default=None, help='Number of chunks to split the data into. Defaults to number of processes.')
    parser.add_argument('--skip_merge', action='store_true', help='Skip merging the chunk files.')
    args = parser.parse_args()

    os.makedirs(args.save_path, exist_ok=True)

    # Determine number of processes and chunks
    num_processes = args.num_processes if args.num_processes else cpu_count()
    num_chunks = args.num_chunks if args.num_chunks else num_processes

    id2ans = load_id2answer(args.raw_CoT_path)
    print(len(id2ans))

    id2reasoning = json.load(open(args.segmented_dataset_path))
    id2reasoning = {str(k): v for k, v in id2reasoning.items()}
    print(len(id2reasoning))

    # id2ans_chunks = split_data_for_parallel(id2ans, num_chunks)
    id2reasoning_chunks = split_data_for_parallel(id2reasoning, num_chunks)

    process_args = []
    for i, id2reasoning_chunk in enumerate(id2reasoning_chunks):
        process_args.append({
            'args': args,
            'chunk_id': i,
            'id2ans': id2ans,
            'id2reasoning': id2reasoning_chunk
        })

    # single_process(process_args[0])
    # return
    # Run processing in parallel
    print("Processing in parallel...")
    with Pool(processes=args.num_processes) as pool:
        results = pool.map(single_process, process_args)
    # Print results
    all_res = {}
    for result in results:
        # print(result)
        all_res.update(result)
    print(f"Processed {len(all_res)} total items")

    # process results
    id2data = load_id2data(args.raw_CoT_path)
    labeled_data = {}
    for k in tqdm(all_res, desc="merging chunks"):
        reasoning_chunks, labels = merge_reasoning_chunks(id2reasoning[k], all_res[k])
        labeled_data[k] = {
            "id": k,
            "question": id2data[k]['instruction'],
            "answer": id2data[k]['answer'],
            "reasoning_chunks": reasoning_chunks,
            "correctness_labels": labels,
        }
        
    # save labeled data
    with open(f'{args.save_path}/labeled_intermediate_answers_{os.path.basename(args.raw_CoT_path)}', 'w') as f:
        for k in labeled_data:
            f.write(json.dumps(labeled_data[k], ensure_ascii=False) + "\n")


    

if __name__ == '__main__':
    main()