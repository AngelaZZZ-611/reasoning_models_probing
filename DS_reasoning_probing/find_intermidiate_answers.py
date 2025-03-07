import torch
import numpy as np
import argparse
import random
import json
import re
import pandas as pd

import nltk
import spacy
from nltk.tokenize import sent_tokenize
from spacy.matcher import Matcher
# Load spaCy English model (ensure you have 'en_core_web_sm' installed)
nlp = spacy.load("en_core_web_sm")
nltk.download('punkt')
nltk.download('punkt_tab')

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


def run_segmentation(args, data_file):
    profile = {}
    for key in data_file.keys():
        if key != 'args':
            data = data_file[key]
            # get the reasoning trace
            reasoning_trace = data['generated_sentence'].split('<think>\n')[1].split('</think>')[0]

            # Split the reasoning trace
            chunks = split_reasoning_trace_with_matcher(reasoning_trace, highlight=args.highlight)
            # # Print results
            # for i, chunk in enumerate(chunks, 1):
            #     print(f"Chunk {i}:\n{chunk}\n")
            print(f"--{key}: {len(chunks)}--")
            # store results
            res = {}
            for i, chunk in enumerate(chunks, 1):
                res[i] = chunk
            profile[key] = res
    return profile



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datafile_path', type=str, default='/scratch/az1658/CoT_explain/20250207_R1_CoT/profile_CoT_generation/raw_CoT/train_dataset_MATH', help='raw CoT file path.')
    parser.add_argument('--save_path', type=str, default='/scratch/az1658/CoT_explain/20250207_R1_CoT/profile_CoT_generation/segemented/train_dataset_MATH', help='save path for the segmented CoTs. ')
    parser.add_argument('--st', type=int, default=0,
                        help='start idx. ')
    parser.add_argument('--highlight', type=bool, default=False,
                        help='whether to highlight the customized transition marker. Just used for debugging purpose. ')
    args = parser.parse_args()


    # load dataset
    ed = 100*args.st+100 
    print(f'==running {100*args.st} to {ed}==')
    data_file = torch.load(f'{args.datafile_path}/CoT_generation_MATH-7474_10000_temp0.6_{100*args.st}_{ed}.pt')

    # run segmentation
    profile = run_segmentation(args, data_file)

    # save
    save_path = f'{args.save_path}/segmented_CoT_MATH-7474_10000_temp0.6_{100*args.st}_{ed}.json'
    with open(save_path, 'w') as fp:
        json.dump(profile, fp)



if __name__ == '__main__':
    main()