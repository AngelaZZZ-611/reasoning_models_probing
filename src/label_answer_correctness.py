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

def single_process(args_dict):
    """Process a single chunk of data."""
    args = args_dict['args']
    chunk_id = args_dict['chunk_id']
    id2ans = args_dict['id2ans']
    id2reasoning = args_dict['id2reasoning']
    assert len(id2ans) == len(id2reasoning), "Length of id2ans and id2reasoning do not match"
    assert not set(id2reasoning.keys()).difference(set(id2ans.keys())), print(set(id2reasoning.keys()).difference(set(id2ans.keys())))

    # load model
    api_key = os.getenv("GEMINI_API_KEY")
    client = genai.Client(api_key=api_key)
    # process prompt and run LLM
    results = {}
    for k in tqdm(id2reasoning.keys()):
        reasoning_trace = id2reasoning[k]
        gt_answer = id2ans[str(k)]
        out_res = run_LLM(args, client, reasoning_trace, gt_answer)

        results[k] = out_res
    
        # print(results)
        # return
    
    # save results
    torch.save(results, f'{args.save_path}/labeled_intermidiate_answers_{os.path.basename(args.raw_CoT_path).strip(".jsonl")}_chunk_{chunk_id}.pt')
    # with open(f'{args.save_path}/labeled_intermidiate_answers_{os.path.basename(args.raw_CoT_path).strip(".jsonl")}_chunk_{chunk_id}.json', 'w') as f:
    #     json.dump(results, f)
    print(f"Processed and saved chunk {chunk_id} with {len(id2ans)} items")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--segmented_dataset_path', type=str, default=None, help='segmented dataset path.')
    parser.add_argument('--raw_CoT_path', type=str, default=None, help='raw CoT dataset path.')
    parser.add_argument('--save_path', type=str, default=None, help='save path.')
    parser.add_argument('--temperature', type=float, default=0.6, 
                        help='temperature setup for generation, the larger the diversity. ')
    parser.add_argument('--num_processes', type=int, default=None, help='Number of processes to use. Defaults to number of CPU cores.')
    parser.add_argument('--num_chunks', type=int, default=None, help='Number of chunks to split the data into. Defaults to number of processes.')
    parser.add_argument('--delete_chunks', action='store_true', help='Delete individual chunk files after merging.')
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

    id2ans_chunks = split_data_for_parallel(id2ans, num_chunks)
    id2reasoning_chunks = split_data_for_parallel(id2reasoning, num_chunks)

    process_args = []
    for i, (id2ans_chunk, id2reasoning_chunk) in enumerate(zip(id2ans_chunks, id2reasoning_chunks)):
        process_args.append({
            'args': args,
            'chunk_id': i,
            'id2ans': id2ans_chunk,
            'id2reasoning': id2reasoning_chunk
        })

    # single_process(process_args[0])
    # return
    # Run processing in parallel
    with Pool(processes=args.num_processes) as pool:
        results = pool.map(single_process, process_args)
    
    # Print results
    all_res = {}
    for result in results:
        # print(result)
        all_res.update(result)
    print(f"Processed {len(all_res)} total items")

    # Save results
    torch.save(all_res, f'{args.save_path}/labeled_intermidiate_answers_{os.path.basename(args.raw_CoT_path).strip(".jsonl")}_merged.pt')

    # remove the intermediate chunk files
    if args.delete_chunks:
        for i in range(num_chunks):
            try:
                file_ = f'{args.save_path}/labeled_intermidiate_answers_{os.path.basename(args.raw_CoT_path).strip(".jsonl")}_chunk_{i}.pt'
                os.remove(file_)
            except Exception as e:
                print(f"Error deleting chunk file {i}: {e}")
    print("All done!")

    

if __name__ == '__main__':
    main()