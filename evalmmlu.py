import json
from tqdm import tqdm
import numpy as np
import re

model = "gpt-5-nano"
dataset = "mmlu_medical"

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

want = "anatomy"
sublist = [d for d in full_outputs if d.get("subject") == want]

print(want)
correct_full = sum([output['correct'] == True for output in sublist])
total_full = len(sublist)
print(f"Correct Full: {correct_full}, Total Full: {total_full}, Accuracy: {correct_full / total_full:.2%}")

want = "clinical_knowledge"
sublist = [d for d in full_outputs if d.get("subject") == want]

print(want)
correct_full = sum([output['correct'] == True for output in sublist])
total_full = len(sublist)
print(f"Correct Full: {correct_full}, Total Full: {total_full}, Accuracy: {correct_full / total_full:.2%}")

want = "college_biology"
sublist = [d for d in full_outputs if d.get("subject") == want]

print(want)
correct_full = sum([output['correct'] == True for output in sublist])
total_full = len(sublist)
print(f"Correct Full: {correct_full}, Total Full: {total_full}, Accuracy: {correct_full / total_full:.2%}")

want = "college_medicine"
sublist = [d for d in full_outputs if d.get("subject") == want]

print(want)
correct_full = sum([output['correct'] == True for output in sublist])
total_full = len(sublist)
print(f"Correct Full: {correct_full}, Total Full: {total_full}, Accuracy: {correct_full / total_full:.2%}")

want = "medical_genetics"
sublist = [d for d in full_outputs if d.get("subject") == want]

print(want)
correct_full = sum([output['correct'] == True for output in sublist])
total_full = len(sublist)
print(f"Correct Full: {correct_full}, Total Full: {total_full}, Accuracy: {correct_full / total_full:.2%}")


want = "professional_medicine"
sublist = [d for d in full_outputs if d.get("subject") == want]

print(want)
correct_full = sum([output['correct'] == True for output in sublist])
total_full = len(sublist)
print(f"Correct Full: {correct_full}, Total Full: {total_full}, Accuracy: {correct_full / total_full:.2%}")
