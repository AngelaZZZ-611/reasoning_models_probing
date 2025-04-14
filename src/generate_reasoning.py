# %%


import torch
import os
import random
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
)
import sys
sys.path.append("../")
from math_eval_tools.main import evaluate as evaluate_math, last_boxed_only_string
from utils import evaluate_mc
import json
from datasets import load_dataset
import numpy as np
from vllm import LLM, SamplingParams

# Global data paths configuration
DATA_PATHS = {
    "gsm8k": "openai/gsm8k",
    "math": "EleutherAI/hendrycks_math",
    "math_500": "HuggingFaceH4/MATH-500",
    "aime_83_24": "di-zhang-fdu/AIME_1983_2024",
    "aime_25": "TIGER-Lab/AIME25",
    "gpqa": "Idavidrein/gpqa"
}

def seed_everything(seed: int):
    import random
    import os
    import numpy as np
    import torch

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True



def generate_answer(model, input_text, temperature=0.7):    
    generate_kwargs = dict(max_new_tokens=32768, temperature=temperature, do_sample=True if temperature > 0.0 else False)
    generated = naive_generation(model, input_text, generate_kwargs)
    return generated
        

def naive_generation(model, input_text, generate_kwargs):
    sampling_param = SamplingParams(
        max_tokens=generate_kwargs.get("max_new_tokens", 512),
        temperature=generate_kwargs.get("temperature", 0.0),
        top_p=generate_kwargs.get("top_p", 0.95),
    )
    outputs = model.generate(input_text, sampling_param)
    response = []
    for output in outputs:
        response.append(output.outputs[0].text)

    if len(response) > 1:
        return response
    return response[0]

def build_gpqa_instruction(example) -> dict:
    ANSWER_LABELS = ["A", "B", "C", "D"]
    PROMPT_PREFIX = "Please choose the correct answer from among the following options: \n"
    # Randomly shuffle the order of the choices every time we generate an exaple
    choice_indices = [1, 2, 3, 4]
    choice_order = random.sample(choice_indices, len(choice_indices))
    ans_idx = choice_order.index(4)

    ordered_choices = [
        example[f"Incorrect Answer {i}"] if i != 4 else example["Correct Answer"]
        for i in choice_order
    ]
    ordered_choices = [
        f"({ANSWER_LABELS[i]}) {choice}" for i, choice in enumerate(ordered_choices)
    ]

    prompt = example["Question"] + "\n" + PROMPT_PREFIX + "\n".join(ordered_choices)
    answer = ANSWER_LABELS[ans_idx]
    example["instruction"] = prompt
    example["answer"] = answer

    return example

def build_prompt(input_text, tokenizer=None, base_model=False, append_str=None):
    if tokenizer is None or not tokenizer.chat_template or base_model:
        input_text_prompt = "Question: " + input_text + "\n" + "Answer: "
    else:
        instruction_following = "\nPlease reason step by step, and put your final answer within \\boxed{}."
        input_text_prompt = tokenizer.apply_chat_template(
            [
                {
                    "role": "user",
                    "content": input_text + instruction_following
                }
            ],
            add_generation_prompt=True,
            tokenize=False
        )
    if append_str is not None:
        input_text_prompt += append_str
    return input_text_prompt

def load_data(data_name="gsm8k", split=None, debug=False):
    if data_name == "gsm8k":
        assert split is not None
        dataset = load_dataset(DATA_PATHS["gsm8k"], split=split)
        list_data_dict = dataset.to_list()
        for item in list_data_dict:
            item["answer"] = item["output"].split("####")[-1].strip()
        if debug:
            list_data_dict = list_data_dict[:10]
        return list_data_dict
    elif data_name == "math":
        new_dataset = []
        for category in ['algebra', 'counting_and_probability', 'geometry', 'intermediate_algebra', 'number_theory', 'prealgebra', 'precalculus']:
            dataset = load_dataset(DATA_PATHS["math"], category, split=split)
            for item in dataset:
                item["instruction"] = item["problem"]
                item["answer"] = last_boxed_only_string(item["solution"])
                if item["answer"] is None:
                    continue
                item.pop("problem")
                new_dataset.append(item)
        if debug:
            new_dataset = new_dataset[:10]
        return new_dataset
    elif data_name == "math_500":
        new_dataset = []
        dataset = load_dataset(DATA_PATHS["math_500"])["test"]
        for item in dataset:
            item["instruction"] = item["problem"]
            item.pop("problem")
            new_dataset.append(item)
        return new_dataset
    elif data_name == "aime_83_24":
        list_of_dict = []
        dataset = load_dataset(DATA_PATHS["aime_83_24"])["train"]
        for item in dataset:
            item["instruction"] = item["Question"]
            item["answer"] = item["Answer"]
            item.pop("Question")
            item.pop("Answer")
            list_of_dict.append(item)
        return list_of_dict 
    elif data_name == "aime_25":
        list_of_dict = []
        dataset = load_dataset(DATA_PATHS["aime_25"])["train"]
        for item in dataset:
            item["instruction"] = item["problem"]
            item.pop("problem")
            list_of_dict.append(item)
        return list_of_dict 
    elif data_name == "gpqa_diamond":
        ds = load_dataset(DATA_PATHS["gpqa"], "gpqa_diamond")["train"]
        ds = ds.map(build_gpqa_instruction)
        ds = ds.select_columns(['instruction', 'answer', 'Correct Answer',"Record ID"])
        return ds.to_list()
    else:
        raise NotImplementedError(f"Dataset {data_name} not supported")

if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="meta-llama/Llama-3.1-8B")
    parser.add_argument("--data_name", type=str, default="math_500")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--start_run", type=int, default=0)
    parser.add_argument("--max_example", type=int, default=-1)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--base_model", action="store_true", help="if true, use `Q: A:` as chat template")
    parser.add_argument("--append_str", type=str, default=None)
    parser.add_argument("--save_path", type=str, default="./initial_cot")
    args = parser.parse_args()

    def convert_json(obj):
        if isinstance(obj, np.float32):
            return float(obj)  # Convert to Python float
        if isinstance(obj, np.int64):
            return int(obj)
        raise TypeError(f"Type {type(obj)} not serializable")

    model_path = args.model_path
    TEMPERATURE = args.temperature

    
    model = LLM(model=model_path, tensor_parallel_size=torch.cuda.device_count(), gpu_memory_utilization=0.95, dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    

    def main(data_name):
        split = None
        if "@" in data_name:
            data_name, split = data_name.split("@")

        if data_name == "gpqa_diamond":
            evaluate_func = evaluate_mc
        else:
            evaluate_func = evaluate_math

        list_data_dict = load_data(data_name, split=split, debug=args.debug)
        if args.debug:
            list_data_dict = list_data_dict[:10]
        if args.max_example != -1:
            list_data_dict = list_data_dict[:args.max_example]

        print(f"Running {len(list_data_dict)} prompts with temperature={TEMPERATURE}")

        model_name = model_path.split("/")[-1]
        data_name_tmp = data_name 
        if data_name_tmp.endswith(".jsonl"):
            data_name_tmp = data_name_tmp.split("/")[-1][:-6]
        setting = f"{model_name}_{data_name_tmp}_rollout_temperature{TEMPERATURE}"
        if split is not None:
            setting = f"{model_name}_{data_name_tmp}-{split}_rollout_temperature{TEMPERATURE}"
        print(f"Setting: {setting}")

        filename = setting
        completions = []
        answers = []
        prompts = []
        for sample in tqdm(list_data_dict):
            input_text = build_prompt(sample["instruction"], tokenizer=tokenizer, base_model=args.base_model, append_str=args.append_str)
            prompts.append(input_text)
            answers.append(sample["answer"])
        
        print("####### Example of Prompts #######")
        print(prompts[0])
        print("##################################")

        completions = generate_answer(model, prompts, temperature=TEMPERATURE)

        eval_results, model_answers, eval_results_summary = evaluate_func(completions=completions, answers=answers)
        # print(eval_results_summary)
        # output_file=f"./initial_rollout/{setting}_results.jsonl"
        # with open(output_file, "a+") as f:
        #     f.write(json.dumps({"setting": setting, "run": run, **eval_results_summary}, default=convert_json) + "\n") 

        # unify id for each example
        global_id = 0
        with open(os.path.join(args.save_path, f"{filename}.jsonl"), "w") as f:
            for completion, answer, res, model_answer, example in zip(completions, answers, eval_results, model_answers, list_data_dict):
                if args.append_str is not None:
                    completion = args.append_str + completion
                if "id" in example:
                    example.pop("id")
                f.write(json.dumps({"completion": completion, "is_correct": res, "extracted_model_answer": model_answer, "id": f"{data_name}-{global_id}", **example}, default=convert_json) + "\n")
                global_id += 1
    os.makedirs(args.save_path, exist_ok=True)
    data_name_list = args.data_name.split(",")
    for data_name in data_name_list:
        main(data_name)
    del model
    
# %%
