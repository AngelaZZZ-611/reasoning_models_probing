import torch
import re
import os
import random
import transformers
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)

from utils import download_url, load_jsonl
import argparse
import json
from .math_grader import execute_grader_with_timeout

transformers.logging.set_verbosity(40)

ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
INVALID_ANS = "[invalid]"

N_SHOT = 8
COT_FLAG = True
DEBUG = False
ANSWER_TRIGGER = "The answer is"
ANSWER_TRIGGER = "Therefore, the answer is"



def extract_answer_from_output(completion):
    match = ANS_RE.search(completion)
    if match:
        match_str = match.group(1).strip()
        match_str = match_str.replace(",", "")
        return match_str
    else:
        return INVALID_ANS


def is_correct(model_answer, answer):
    # gt_answer = extract_answer_from_output(answer)
    execution_result = execute_grader_with_timeout(model_answer, answer)
    if execution_result is not None:
        return execution_result
    else:
        return False

    # gt_answer = extract_answer_from_output(answer)
    # assert gt_answer != INVALID_ANS
    # return model_answer == gt_answer

def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    left_brace_idx = None
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
            if left_brace_idx is None:
                left_brace_idx = i
        elif string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break

        i += 1
    
    if left_brace_idx is None or right_brace_idx is None:
        return None

    return string[left_brace_idx + 1: right_brace_idx].strip()

def clean_answer(model_pred):
    

    match_str = last_boxed_only_string(model_pred)
    if match_str is not None:
        return match_str

    model_pred = model_pred.lower()
    preds = model_pred.split(ANSWER_TRIGGER.lower())
    answer_flag = True if len(preds) > 1 else False
    if answer_flag:
        # Pick first answer with flag
        pred = preds[1]
    else:
        # Pick last number without flag
        pred = preds[-1]

    pred = pred.replace(",", "")
    pred = [s for s in re.findall(r"-?\d+\.?\d*", pred)]

    if len(pred) == 0:
        return INVALID_ANS

    if answer_flag:
        # choose the first element in list
        pred = pred[0]
    else:
        # choose the last element in list
        pred = pred[-1]

    # (For arithmetic tasks) if a word ends with period, it will be omitted ...
    if pred[-1] == ".":
        pred = pred[:-1]

    return pred


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


def load(model_name_or_path):
    print(f"Loading model from {model_name_or_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        else:
            tokenizer.pad_token_id = 0

    model.eval()

    return model, tokenizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="/dataset/llama2/llama-2-7b-hf",
        help="The model checkpoint for weights initialization.",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="./data",
        help="The root folder of the data.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output",
        help="The output directory where the model predictions and checkpoints will be written.",
    )

    parser.add_argument("--load", type=str, default=None, help="load quantized model")

    args = parser.parse_args()
    return args


def generate(model, tokenizer, input_text, generate_kwargs):
    input_text = tokenizer(
        input_text,
        padding=False,
        add_special_tokens=True,
        return_tensors="pt",
    )
    input_ids = input_text.input_ids.cuda()
    attention_mask = input_text.attention_mask.cuda()

    output_ids = model.generate(
        input_ids=input_ids, attention_mask=attention_mask, **generate_kwargs
    )
    response = []
    for i in range(output_ids.shape[0]):
        response.append(
            tokenizer.decode(
                output_ids[i][input_ids.shape[1] :],
                skip_special_tokens=True,
                ignore_tokenization_space=True,
            )
        )

    if len(response) > 1:
        return response
    return response[0]


def main():
    args = parse_args()

    seed_everything(args.seed)

    test_filepath = os.path.join(args.data_root, "gsm8k_test.jsonl")
    if not os.path.exists(test_filepath):
        download_url(
            "https://raw.githubusercontent.com/openai/"
            "grade-school-math/2909d34ef28520753df82a2234c357259d254aa8/"
            "grade_school_math/data/test.jsonl",
            args.data_root,
        )
        os.rename(os.path.join(args.data_root, "test.jsonl"), test_filepath)

    list_data_dict = load_jsonl(test_filepath, instruction="question", output="answer")

    model, tokenizer = load(args.model_name_or_path)

    if args.load:
        print("loading...", args.load)
        model_state = torch.load(args.load, map_location="cpu")
        model.load_state_dict(model_state, strict=False)
        model.half().cuda()

    answers = []
    for sample in tqdm(list_data_dict):
        input_text = build_prompt(sample["instruction"], N_SHOT, COT_FLAG)
        generate_kwargs = dict(max_new_tokens=256, top_p=0.95, temperature=0.8)
        model_completion = generate(model, tokenizer, input_text, generate_kwargs)
        model_answer = clean_answer(model_completion)
        is_cor = is_correct(model_answer, sample["output"])
        answers.append(is_cor)
        if DEBUG:
            print(f"Full input_text:\n{input_text}\n\n")
        print(
            f'Question: {sample["instruction"]}\n\n'
            f'Answers: {extract_answer_from_output(sample["output"])}\n\n'
            f"Model Answers: {model_answer}\n\n"
            f"Model Completion: {model_completion}\n\n"
            f"Is correct: {is_cor}\n\n"
        )

        print(
            f"Num of total question: {len(answers)}, "
            f"Correct num: {sum(answers)}, "
            f"Accuracy: {float(sum(answers))/len(answers)}."
        )

    os.makedirs(args.output_dir, exist_ok=True)

    with open(os.path.join(args.output_dir, "results.txt"), "w") as f:
        for answer in answers:
            print(answer, file=f)

    with open(os.path.join(args.output_dir, "scores.txt"), "w") as f:
        print(
            f"Num of total question: {len(answers)}, "
            f"Correct num: {sum(answers)}, "
            f"Accuracy: {float(sum(answers))/len(answers)}.",
            file=f,
        )

def evaluate(filepath=None, answers = None, completions = None, output_file=None):
    # args = parse_args()

    # seed_everything(args.seed)

    # test_filepath = os.path.join("./data", "gsm8k_test.jsonl")
    # if not os.path.exists(test_filepath):
    #     download_url(
    #         "https://raw.githubusercontent.com/openai/"
    #         "grade-school-math/2909d34ef28520753df82a2234c357259d254aa8/"
    #         "grade_school_math/data/test.jsonl",
    #         "./data",
    #     )
    #     os.rename(os.path.join("./data", "test.jsonl"), test_filepath)

    # list_data_dict = load_jsonl(test_filepath, instruction="question", output="answer")

    eval_results = []
    model_answers = []
    for answer, model_completion in zip(answers, completions):
        model_answer = clean_answer(model_completion)
        is_cor = is_correct(model_answer, answer)
        eval_results.append(is_cor)
        model_answers.append(model_answer)
        # if DEBUG:
        #     print(f"Full input_text:\n{input_text}\n\n")
        # print(
        #     f'Question: {sample["instruction"]}\n\n'
        #     f'Answers: {extract_answer_from_output(sample["output"])}\n\n'
        #     f"Model Answers: {model_answer}\n\n"
        #     f"Model Completion: {model_completion}\n\n"
        #     f"Is correct: {is_cor}\n\n"
        # )

    # print(
    #     f"Num of total question: {len(eval_results)}, "
    #     f"Correct num: {sum(eval_results)}, "
    #     f"Accuracy: {float(sum(eval_results))/len(eval_results)}."
    # )

    # if output_file is not None:
    #     os.makedirs(os.path.dirname(output_file), exist_ok=True)
    #     with open(output_file, "w") as f:
    #         # for answer, model_completion, is_cor in zip(answers, completions, eval_results):
    #         f.write(json.dumps({"total": len(eval_results), "correct_num": sum(eval_results), "accuracy": float(sum(eval_results))/len(eval_results)}) + "\n")

    # os.makedirs(args.output_dir, exist_ok=True)

    # with open(os.path.join(args.output_dir, "results.txt"), "w") as f:
    #     for answer in answers:
    #         print(answer, file=f)

    # with open(os.path.join(args.output_dir, "scores.txt"), "w") as f:
    #     print(
    #         f"Num of total question: {len(answers)}, "
    #         f"Correct num: {sum(answers)}, "
    #         f"Accuracy: {float(sum(answers))/len(answers)}.",
    #         file=f,
    #     )
    return eval_results, model_answers, {"total": len(eval_results), "correct_num": sum(eval_results), "accuracy": float(sum(eval_results))/len(eval_results)}

if __name__ == "__main__":
    main()
