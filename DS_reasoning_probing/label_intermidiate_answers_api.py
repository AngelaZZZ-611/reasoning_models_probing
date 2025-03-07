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

# Set your OpenAI API key as an environment variable 
os.environ["GEMINI_API_KEY"] = "" #"your_api_key_here" 



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
    # output = response.text
    # # convert the JSON formatted text string into a JSON object
    # json_string = output.replace('```json\n', '').replace('\n```', '')
    # json_object = json.loads(json_string)
    # return json_object
    return response



# response.text = '```json\n[\n  {"id": "1", "result": "5/3", "correctness": null},\n  {"id": "2", "result": "14/3", "correctness": true},\n  {"id": "3", "result": "14/3", "correctness": true},\n  {"id": "4", "result": null, "correctness": null},\n  {"id": "5", "result": null, "correctness": null},\n  {"id": "6", "result": null, "correctness": null},\n  {"id": "7", "result": "14/3", "correctness": true},\n  {"id": "8", "result": null, "correctness": null},\n  {"id": "9", "result": "\\\\dfrac{14}{3}", "correctness": true}\n]\n```'



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--segmented_dataset_path', type=str, default='/scratch/az1658/CoT_explain/20250207_R1_CoT/profile_CoT_generation/segemented/train_dataset_MATH', help='segmented dataset path.')
    parser.add_argument('--raw_CoT_path', type=str, default='/scratch/az1658/CoT_explain/20250207_R1_CoT/profile_CoT_generation/raw_CoT/train_dataset_MATH', help='raw CoT dataset path.')
    parser.add_argument('--save_path', type=str, default='/scratch/az1658/CoT_explain/20250207_R1_CoT/profile_CoT_generation/label_intermediate_answers/train_dataset_MATH', help='save path.')
    parser.add_argument('--temperature', type=float, default=0.6, 
                        help='temperature setup for generation, the larger the diversity. ')
    # parser.add_argument('--top_p', type=float, default=0.99,
    #                     help='top_p setup for generation, the larger the diversity. ')
    parser.add_argument('--st', type=int, default=0,
                        help='start idx.')
    args = parser.parse_args()


    # load model
    api_key = os.getenv("GEMINI_API_KEY")
    client = genai.Client(api_key=api_key)

    # load segmented data-files
    ed = 100*args.st+100
    print(f'==running {100*args.st} to {ed}==')
    file_path = f'{args.segmented_dataset_path}/segmented_CoT_MATH-7474_10000_temp0.6_{100*args.st}_{ed}.json'
    with open(file_path, 'r') as f:
        data = json.load(f)
    # load file to get the gt-answer
    file_CoT_gen = torch.load(f'{args.raw_CoT_path}/CoT_generation_MATH-7474_10000_temp0.6_{100*args.st}_{ed}.pt')

    # process prompt and run LLM
    results = {}
    for k in data.keys():
        reasoning_trace = data[k]
        gt_answer = file_CoT_gen[int(k)]['gt_answer']
        out_res = run_LLM(args, client, reasoning_trace, gt_answer)
        results[k] = out_res
    
    # save results
    torch.save(results, f'{args.save_path}/labeled_intermidiate_answers_CoT_MATH-7474_10000_temp0.6_{100*args.st}_{ed}.pt')


if __name__ == '__main__':
    main()