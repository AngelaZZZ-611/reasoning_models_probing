# %%


import torch
import os
import random
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
)
from math_eval_tools.main import evaluate as evaluate_math, last_boxed_only_string
from utils import evaluate_mc
import json
from datasets import load_dataset
import numpy as np

def load_jsonl(
    file_path,
    instruction="instruction",
    input="input",
    output="output",
    category="category",
    is_gzip=False,
):
    # Format of each line:
    # {'instruction': ..., 'input': ..., 'output':...}
    list_data_dict = []
    open_func = open if not is_gzip else gzip.open
    with open_func(file_path, "r") as f:
        for line in f:
            item = json.loads(line)
            new_item = dict(
                instruction=item[instruction] if instruction in item else None,
                input=item[input] if input in item else None,
                output=item[output] if output in item else None,
                category=item[category] if category in item else None,
            )
            item = new_item
            list_data_dict.append(item)
    return list_data_dict


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

def concat_text(text_a, text_b):
    if isinstance(text_a, list):
        assert isinstance(text_b, list)
        return [a + " " + b.strip() + " " for a, b in zip(text_a, text_b)]
    return text_a + " " + text_b.strip() + " "

def generate_answer(model, input_text, temperature=0.7, full_r1=False):
    ###############
    if full_r1:
        generation_func = naive_r1_generation
    else:
        generation_func = naive_generation
    #################

    
    generate_kwargs = dict(max_new_tokens=5000, temperature=temperature, do_sample=True if temperature > 0.0 else False)
    generated = generation_func(model, input_text, generate_kwargs)
    return generated


def naive_r1_generation(model, input_text, generate_kwargs):
    completions = []
    answer_logprobs = []
    for prompt in tqdm(input_text):
        # outputs = json.load(open("./output.json"))
        if "max_new_tokens" in generate_kwargs:
            generate_kwargs.pop("max_new_tokens")
        outputs = model(prompt, 
                max_tokens=3276800, 
                stop=["<｜User｜>"], 
                echo=False, 
                logprobs=generate_kwargs.get("logprobs", 0), temperature=generate_kwargs.get("temperature", 0.0),
                top_p=generate_kwargs.get("top_p", 0.95), top_k=generate_kwargs.get("top_k", 50))
        completions.append(outputs["choices"][0]["text"])
        if outputs["choices"][0].get("logprobs", None):
            answer_logprobs.append(outputs["choices"][0]["logprobs"])
    return completions, answer_logprobs


        

def naive_generation(model, input_text, generate_kwargs):
    sampling_param = SamplingParams(
        max_tokens=generate_kwargs.get("max_new_tokens", 512),
        temperature=generate_kwargs.get("temperature", 0.0),
        top_p=generate_kwargs.get("top_p", 0.95),
    )
    # sampling_param = SamplingParams(
    #     temperature=generate_kwargs.get("temperature", 0.0),
    #     top_k=generate_kwargs.get("top_k", 50),
    #     top_p=generate_kwargs.get("top_p", 1.0),
    #     max_tokens=generate_kwargs.get("max_new_tokens", 512),
    #     logprobs = generate_kwargs.get("logprobs", 0)
    # )
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
    # question = PROMPT_SUFFIX
    answer = ANSWER_LABELS[ans_idx]
    example["instruction"] = prompt
    example["answer"] = answer

    return example

def build_prompt(input_text, tokenizer=None, cot=False, base_model=False, append_str=None, full_r1=False):
    if full_r1:
        input_text_prompt = f"<｜User｜>{input_text} Please reason step by step, and put your final answer within \\boxed{{}}.<｜Assistant｜><think>\n"
    elif tokenizer is None or not tokenizer.chat_template or base_model:
        input_text_prompt = "Question: " + input_text + "\n" + "Answer: "
    else:
        input_text_prompt = tokenizer.apply_chat_template(
            [
                {
                    "role": "system",
                    # "content": "You are a helpful and harmless assistant. You are Qwen developed by Alibaba. You should think step-by-step."
                    "content": "Please reason step by step, and put your final answer within \\boxed{{}}."
                },
                {
                "role": "user",
                "content": input_text
            }
            ],
            add_generation_prompt=True,
            tokenize=False
        )
    # if cot:
    #     input_text_prompt += "Let's think step by step. "
    # if add_think_tokens:
    #     input_text_prompt += " <think>\n</think>\n"
    if append_str is not None:
        input_text_prompt += append_str
    return input_text_prompt

def load_data(data_name="gsm8k", split=None, debug=False):
    if data_name == "gsm8k":
        assert split is not None
        test_filepath = os.path.join("/home/test/test05/cyl/moe_eval_dataset/gsm8k", f"{split}.jsonl")
        # if not os.path.exists(test_filepath):
        #     download_url(
        #         "https://raw.githubusercontent.com/openai/"
        #         "grade-school-math/2909d34ef28520753df82a2234c357259d254aa8/"
        #         "grade_school_math/data/test.jsonl",
        #         "./data",
        #     )
        #     os.rename(os.path.join("./data", "test.jsonl"), test_filepath)

        list_data_dict = load_jsonl(test_filepath, instruction="question", output="answer")
        for item in list_data_dict:
            item["answer"] = item["output"].split("####")[-1].strip()
        if debug:
            list_data_dict = list_data_dict[:10]
        return list_data_dict
    elif data_name == "math":
        new_dataset = []
        for category in ['algebra', 'counting_and_probability', 'geometry', 'intermediate_algebra', 'number_theory', 'prealgebra', 'precalculus']:
            dataset = load_dataset("/scratch/yc7320/hf_dataset/MATH", category, split=split)
            for item in dataset:
                item["instruction"] = item["problem"]
                item["answer"] = last_boxed_only_string(item["solution"])
                if item["answer"] is None:
                    continue
                # assert item["answer"] is not None, print(item["solution"])
                item.pop("problem")
                new_dataset.append(item)
        if debug:
            new_dataset = new_dataset[:10]
        return new_dataset
    elif data_name == "math_500":
        raise NotImplementedError
        new_dataset = []
        dataset = load_dataset("HuggingFaceH4/MATH-500")["test"]
        for item in dataset:
            item["instruction"] = item["problem"]
            item.pop("problem")
            new_dataset.append(item)
        return new_dataset
    elif data_name == "aime_2024":
        raise NotImplementedError
        new_dataset = []
        dataset = load_dataset("HuggingFaceH4/aime_2024")["train"]
        for item in dataset:
            item["instruction"] = item["problem"]
            item.pop("problem")
            new_dataset.append(item)
        return new_dataset
    # elif data_name.endswith(".jsonl"):
    #     with open(data_name) as f:
    #         list_data_dict = [json.loads(line) for line in f]                
    #     for item in list_data_dict:
    #         for k in ["completion", "is_correct", "top_logprobs", "extracted_model_answer"]:
    #             if k in item:
    #                 item.pop(k)
    #     return list_data_dict
    elif data_name == "gpqa_diamond":
        raise NotImplementedError
        ds = load_dataset("./data/gpqa", "gpqa_diamond")["train"]
        ds = ds.map(build_gpqa_instruction)
        ds = ds.select_columns(['instruction', 'answer', 'Correct Answer',"Record ID"])
        return ds.to_list()
    else:
        raise NotImplementedError

if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="/home/test/test05/cyl/models/Llama-3.1-8B")
    parser.add_argument("--data_name", type=str, default="math_500")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--run_number", type=int, default=1, help="Number of runs")
    parser.add_argument("--start_run", type=int, default=0)

    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--base_model", action="store_true", help="if true, use `Q: A:` as chat template")
    parser.add_argument("--append_str", type=str, default=None)


    args = parser.parse_args()

    def convert_json(obj):
        if isinstance(obj, np.float32):
            return float(obj)  # Convert to Python float
        if isinstance(obj, np.int64):
            return int(obj)
        raise TypeError(f"Type {type(obj)} not serializable")

    model_path = args.model_path
    TEMPERATURE = args.temperature
    RUN_NUMBER = args.run_number

    # model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map="auto")
    full_r1 = False
    if "deepseek" in model_path.lower() and "distill" not in model_path.lower():
        # use llamacpp to load full r1
        raise NotImplementedError
        
    else:
        from vllm import LLM, SamplingParams

        model = LLM(model=model_path, tensor_parallel_size=torch.cuda.device_count(), gpu_memory_utilization=0.95, dtype=torch.float16)
        # tensor_parallel_size=torch.cuda.device_count(), 
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

        print(f"Running {len(list_data_dict)} prompts with temperature={TEMPERATURE}, total {RUN_NUMBER} runs")

        model_name = model_path.split("/")[-1]
        data_name_tmp = data_name 
        if data_name_tmp.endswith(".jsonl"):
            data_name_tmp = data_name_tmp.split("/")[-1][:-6]
        setting = f"{model_name}_{data_name_tmp}_rollout_temperature{TEMPERATURE}"
        if split is not None:
            setting = f"{model_name}_{data_name_tmp}-{split}_rollout_temperature{TEMPERATURE}"
        print(f"Setting: {setting}")
        for run in range(args.start_run, RUN_NUMBER):
            seed_everything(run)
            if split is None:
                filename = f"{setting}_run{run}"
            else:
                filename = f"{setting}_run{run}"
            print(f"Running {run}th run: {filename}")

            completions = []
            answers = []
            prompts = []
            for sample in tqdm(list_data_dict):
                input_text = build_prompt(sample["instruction"], tokenizer=tokenizer, full_r1=full_r1, base_model=args.base_model, append_str=args.append_str)
                prompts.append(input_text)
                answers.append(sample["answer"])
            
            print("####### Example of Prompts #######")
            print("\n".join(prompts[0:5]))
            print("##################################")

            completions = generate_answer(model, prompts, temperature=TEMPERATURE, full_r1 = full_r1)

            eval_results, model_answers, eval_results_summary = evaluate_func(completions=completions, answers=answers)
            print(eval_results_summary)
            output_file=f"./initial_rollout/{setting}_results.jsonl"
            with open(output_file, "a+") as f:
                f.write(json.dumps({"setting": setting, "run": run, **eval_results_summary}, default=convert_json) + "\n") 

            
            with open(f"./initial_rollout/{filename}.jsonl", "w") as f:
                for completion, answer, extracted_answer, res, example in zip(completions, answers, model_answers, eval_results, list_data_dict):
                    f.write(json.dumps({"completion": completion, "top_logprobs": None, "extracted_model_answer": extracted_answer, "is_correct": res, **example}, default=convert_json) + "\n")
    
    data_name_list = args.data_name.split(",")
    for data_name in data_name_list:
        main(data_name)
    del model
    
# %%
