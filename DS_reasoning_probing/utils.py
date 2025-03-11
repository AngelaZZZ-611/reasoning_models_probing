import copy
import os
import ssl
import urllib.request

import os.path as osp
import gzip
import json
import re

ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
INVALID_ANS = "[invalid]"
ANSWER_TRIGGER = "The answer is"

def download_url(url: str, folder="folder"):
    """
    Downloads the content of an url to a folder. Modified from \
    https://github.com/pyg-team/pytorch_geometric/tree/master/torch_geometric

    Args:
        url (string): The url of target file.
        folder (string): The target folder.

    Returns:
        string: File path of downloaded files.
    """

    file = url.rpartition("/")[2]
    file = file if file[0] == "?" else file.split("?")[0]
    path = osp.join(folder, file)
    if osp.exists(path):
        print(f"File {file} exists, use existing file.")
        return path

    print(f"Downloading {url}")
    os.makedirs(folder, exist_ok=True)
    ctx = ssl._create_unverified_context()
    data = urllib.request.urlopen(url, context=ctx)
    with open(path, "wb") as f:
        f.write(data.read())

    return path


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

def evaluate_mc(completions, answers):
    """
    Evaluate multiple-choice completions.
    Args:
        completions (List[str]): List of completions.
        answers (List[str]): List of answers.
    Returns:
        float: Accuracy.
    """

    # extract answer from the completion
    patterns = [r'answer is \((.)\)', r'Answer: \((.)\)', r'answer: \((.)\)', r'answer \((.)\)', r'\((.)\)']
    pred_list = []
    format_error = 0
    for pred in completions:
        matched_pred = last_boxed_only_string(pred)

        if matched_pred is None:
            last_match = None
            last_index = -1  # Track the last occurrence index

            for pattern in patterns:
                for match in re.finditer(pattern, pred):  # Iterate over all matches for the pattern
                    if match and match.group(1):
                        start_index = match.start()  # Get the starting position of the match
                        if start_index > last_index:  # Update if it's the latest match
                            last_index = start_index
                            last_match = match.group(1)

            if last_match:
                matched_pred = last_match
            else:
                format_error += 1  # No match found

        if matched_pred:
            letter_match = re.search(r'(?i)\b[A-D]\b', matched_pred)
            if letter_match:
                matched_pred = letter_match.group(0).upper()
        
        pred_list.append(matched_pred)
    
    assert len(pred_list) == len(answers)
    
    correct = 0
    total = 0
    eval_results = []
    for pred, ans in zip(pred_list, answers):
        if pred == ans:
            correct += 1
        total += 1
        eval_results.append(int(pred == ans))
    return eval_results, pred_list, {
        "accuracy": correct / len(pred_list),
        "correct": correct,
        "total": len(pred_list),
        "format_error": format_error
    }


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


if __name__ == "__main__":
    import json
    filepath = "/scratch/yc7320/o1_analysis/results/DeepSeek-R1-Distill-Qwen-32B_gpqa_diamond_elicit_1turn_output_0.7temp_run0.jsonl"
    data = [json.loads(line) for line in open(filepath, "r").readlines()]

    completions = [item["completion"] for item in data]
    outputs = [c.split("</think>")[-1].strip() if "</think>" in c else c for c in completions]
    answers = [item["answer"] for item in data]
    eval_results, pred_list, eval_dict = evaluate_mc(outputs, answers)
    for res, item, pred in zip(eval_results, data, pred_list):
        item["extracted_model_answer"]  = pred
        item["is_correct"] = res
    print(eval_dict)
    with open(filepath.replace(".jsonl", "_neweval.jsonl"), "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")
