import torch
import numpy as np
import argparse
import random
import json
import re
import pandas as pd
import os
# from find_intermidiate_answers import split_reasoning_trace_with_matcher

import nltk
import spacy
from nltk.tokenize import sent_tokenize
from spacy.matcher import Matcher
import json
import glob
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
# Load spaCy English model (ensure you have 'en_core_web_sm' installed)
nlp = spacy.load("./en_core_web_sm-3.8.0/en_core_web_sm/en_core_web_sm-3.8.0")
# nltk.download('punkt')
# nltk.download('punkt_tab')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)
# np.random.seed(0)
random.seed(0)

# Define custom transition markers
custom_transitions = {"wait", "double-check", "alternatively", "make sure", "another way",
                      "verify", "to confirm"}
# custom_transitions = {"answer", "result", "value"} # no, not good performance, large error-rate.

# Create a Matcher for detecting multi-word expressions
matcher = Matcher(nlp.vocab)

# Add multi-word patterns to the matcher
for phrase in custom_transitions:
    pattern = [{"LOWER": word} for word in phrase.split()]
    matcher.add(phrase, [pattern])  # Add pattern to matcher

def split_reasoning_trace_with_matcher(text, highlight=True):
    # Tokenize text into sentences
    # sentences = sent_tokenize(text)
    # first split by newline ("\n\n") instead of sentence, because DS ususally takes newline as a step.
    steps = text.split("\n\n") 
    chunks = []
    current_chunk = []
    # for sentence in sentences:
    for sentence in steps:
        doc = nlp(sentence)  # Process with spaCy
        matches = matcher(doc)  # Find matches in the sentence
        transition_found = None
        for match_id, start, end in matches:
            matched_text = doc[start:end].text  # Get the matched phrase
            transition_found = matched_text
            break  # We only need the first transition match per sentence
        # If a transition phrase is found, start a new chunk
        if transition_found:
            if current_chunk:
                # chunks.append(" ".join(current_chunk))  # Save previous chunk 
                chunks.append("\n\n".join(current_chunk)) # if above is splitted by newline initially. 
            # Optionally highlight the transition phrase
            marked_sentence = sentence.replace(transition_found, f"**{transition_found}**") if highlight else sentence
            current_chunk = [marked_sentence]  # Start a new chunk
        else:
            current_chunk.append(sentence)
    # Add the last chunk if it exists
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks

def run_segmentation(args_dict):
    """Process a single chunk of data."""
    args = args_dict['args']
    chunk_id = args_dict['chunk_id']
    chunk_data = args_dict['chunk_data']

    profile = {}
    for key in tqdm(chunk_data.keys()):
        # get the reasoning trace
        if "<think>" in chunk_data[key]:
            reasoning_trace = chunk_data[key].split('<think>\n')[1].split('</think>')[0]
        else:
            if "</think>" not in chunk_data[key]:
                print(f"Key {key} does not contain a proper reasoning trace.")
            reasoning_trace = chunk_data[key].split("</think>")[0]

        # Split the reasoning trace
        chunks = split_reasoning_trace_with_matcher(reasoning_trace, highlight=args.highlight)
        print(f"--{key}: {len(chunks)}--")
        # store results
        res = {}
        for i, chunk in enumerate(chunks, 1):
            res[i] = chunk
        profile[key] = res
    
    # Save the result for this chunk
    chunk_filename = f'segmented_CoT_{os.path.basename(args.datafile_path).strip(".jsonl")}_chunk_{chunk_id}.json'
    
    save_path = os.path.join(args.save_path, chunk_filename)
    with open(save_path, 'w') as fp:
        json.dump(profile, fp)
    
    return f"Processed and saved chunk {chunk_id} with {len(chunk_data)} items"

def load_data_file(datafile_path):
    # load it to {"id": reasoning}
    if datafile_path.endswith("jsonl"):
        data = {}
        i = 0
        add_id = False
        with open(datafile_path, 'r') as f:
            json_data = [json.loads(line) for line in f.readlines()]
            for line in json_data:
                if "id" not in line:
                    add_id = True
                    line["id"] = i
                    i += 1
                data[line["id"]] = line['completion']
        if add_id:
            with open(datafile_path, 'w') as f:
                for line in json_data:
                    f.write(json.dumps(line) + '\n')
    elif datafile_path.endswith("pt"):
        raise NotImplementedError
    return data

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

def merge_chunk_files(save_path, datafile_path, delete_chunks=False):
    """Merge all chunk files into a single file."""
    base_filename = os.path.basename(datafile_path).strip(".jsonl")
    pattern = os.path.join(save_path, f'segmented_CoT_{base_filename}_chunk_*.json')
    chunk_files = glob.glob(pattern)
    
    if not chunk_files:
        print(f"No chunk files found matching pattern: {pattern}")
        return
    
    merged_data = {}
    for file_path in chunk_files:
        try:
            with open(file_path, 'r') as f:
                chunk_data = json.load(f)
                merged_data.update(chunk_data)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
    
    merged_file_path = os.path.join(save_path, f'segmented_CoT_{base_filename}_merged.json')
    with open(merged_file_path, 'w') as f:
        json.dump(merged_data, f)
    
    print(f"Merged {len(chunk_files)} chunk files into {merged_file_path} with {len(merged_data)} total items")
    
    # Optionally delete individual chunk files
    if delete_chunks:
        for file_path in chunk_files:
            try:
                os.remove(file_path)
                print(f"Deleted chunk file: {file_path}")
            except Exception as e:
                print(f"Error deleting {file_path}: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datafile_path', type=str, default='/scratch/az1658/CoT_explain/20250207_R1_CoT/profile_CoT_generation/raw_CoT/train_dataset_MATH', help='raw CoT file path.')
    parser.add_argument('--save_path', type=str, default='/scratch/az1658/CoT_explain/20250207_R1_CoT/profile_CoT_generation/segemented/train_dataset_MATH', help='save path for the segmented CoTs.')
    parser.add_argument('--num_processes', type=int, default=None, help='Number of processes to use. Defaults to number of CPU cores.')
    parser.add_argument('--num_chunks', type=int, default=None, help='Number of chunks to split the data into. Defaults to number of processes.')
    parser.add_argument('--highlight', type=bool, default=False, help='whether to highlight the customized transition marker. Just used for debugging purpose.')
    parser.add_argument('--delete_chunks', action='store_true', help='Delete individual chunk files after merging.')
    parser.add_argument('--skip_merge', action='store_true', help='Skip merging the chunk files.')
    args = parser.parse_args()

    # Ensure save directory exists
    os.makedirs(args.save_path, exist_ok=True)

    # Load dataset
    data_file = load_data_file(args.datafile_path)
    
    # Determine number of processes and chunks
    num_processes = args.num_processes if args.num_processes else cpu_count()
    num_chunks = args.num_chunks if args.num_chunks else num_processes
    
    print(f"Splitting data into {num_chunks} chunks for processing with {num_processes} processes")
    
    # Split data into chunks
    chunks = split_data_for_parallel(data_file, num_chunks)
    
    # Prepare arguments for each worker process
    process_args = []
    for i, chunk_data in enumerate(chunks):
        process_args.append({
            'args': args,
            'chunk_id': i,
            'chunk_data': chunk_data
        })
    
    # Run processing in parallel
    with Pool(processes=num_processes) as pool:
        results = pool.map(run_segmentation, process_args)
    
    # Print results
    for result in results:
        print(result)
    
    # Merge the chunk files if not skipped
    if not args.skip_merge:
        print("Merging chunk files...")
        merge_chunk_files(args.save_path, args.datafile_path, delete_chunks=args.delete_chunks)

if __name__ == "__main__":
    main()