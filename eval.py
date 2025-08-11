import json
from tqdm import tqdm
import numpy as np
import re

model = "gpt-4o-2024-11-20"
dataset = "medqa"

split_list = ["text"]

full_outputs = []

for split in split_list:

    prompting_type = "cot"

    result_path = f"outputs/dev/{model}/{dataset}/zero_shot/{prompting_type}/{dataset}_{split}_output.jsonl"

    with open(result_path, "r") as f:
        outputs = [json.loads(line) for line in f]

    # Data Check
    source_path = f"data/{dataset}/input/{dataset}_{split}_input.jsonl"
    with open(source_path, "r") as f:
        sources = [json.loads(line) for line in f]
    assert len(sources) == len(outputs)
    for i, source in enumerate(sources):
        assert source['id'] == outputs[i]['id']
        assert source['question'] == outputs[i]['question']

    print(f"Loaded {len(outputs)} outputs")
    full_outputs.extend(outputs)

print(f"Loaded {len(full_outputs)} full outputs")

def split_string(s):
    parts = re.split(r'(?i)final answer', s)
    return parts

# Results stats
print("------------------- Main Results -------------------")
print(f"Model: {model}")

print()
correct_full = sum([output['correct'] == True for output in full_outputs])
total_full = len(full_outputs)
print(f"Correct Full: {correct_full}, Total Full: {total_full}, Accuracy: {correct_full / total_full:.2%}")