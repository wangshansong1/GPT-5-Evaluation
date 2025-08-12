import json
from tqdm import tqdm
import numpy as np
import re

# gpt-4o-2024-11-20, claude-3-5-sonnet-20241022, gemini-1.5-pro, gpt-4o-mini, deepseek-reasoner...
model = "gpt-5-mini"

dataset = "medxpertqa"

split_list = ["mm"]

full_outputs = []

for split in split_list:

    if "qvq" in model.lower():
        prompting_type = "ao"
    else:
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

if "qvq" in model.lower():
    print(model)
    new_data = []
    for index, line in enumerate(tqdm(full_outputs)):
        prediction_rationale = line["messages"][-1]["content"]

        if re.search(r'(?i)final answer', prediction_rationale):
            flag = True
        else:
            flag = False

        prediction = split_string(prediction_rationale)[-1].strip()

        if line['id'].lower().startswith("text"):
            u_pattern = r"[A-J]"
            l_pattern = r"[a-j]"
        else:
            u_pattern = r"[A-E]"
            l_pattern = r"[a-e]"

        letter_match = re.findall(u_pattern, prediction)
        if letter_match:
            if flag:
                prediction = letter_match[0]
            else:
                prediction = letter_match[-1]
        else:
            letter_match = re.findall(l_pattern, prediction)
            if letter_match:
                if flag:
                    prediction = letter_match[0].upper()
                else:
                    prediction = letter_match[-1].upper()

        label = line["label"][0]
        line["prediction"] = prediction
        line["correct"] = prediction == label
        new_data.append(line)
    full_outputs = new_data
elif model == "deepseek-reasoner":
    print(model)
    new_data = []
    for index, line in enumerate(tqdm(full_outputs)):
        assert "Put your final" in line['messages'][-2]["content"]

        prediction = line['response']

        if line['id'].lower().startswith("text"):
            pattern = r"\\boxed{([A-J])}"
        else:
            pattern = r"\\boxed{([A-E])}"

        letter_match = re.findall(pattern, prediction)
        prediction = letter_match[0] if letter_match else prediction
        label = line['label'][0]
        line["prediction"] = prediction
        line["correct"] = prediction == label
        new_data.append(line)
    full_outputs = new_data

    # Results stats
types = set([output['question_type'] for output in full_outputs])
types = sorted(list(types))
print(f"Types: {types}")

print("------------------- Main Results -------------------")
print(f"Model: {model}")

print()
for split in split_list:
    for type in types:
        correct = sum([output['correct'] == True for output in full_outputs if output['question_type'] == type and output['id'].lower().startswith(split)])
        total = sum([output['question_type'] == type and output['id'].lower().startswith(split) for output in full_outputs])
        print(f"Split: {split}, Type: {type}, Correct: {correct}, Total: {total}, Accuracy: {correct/total:.2%}")
    correct = sum([output['correct'] == True for output in full_outputs if output['id'].lower().startswith(split)])
    total = sum([output['id'].lower().startswith(split) for output in full_outputs])
    print(f"Split: {split}, Correct: {correct}, Total: {total}, Accuracy: {correct / total:.2%}")
    print()

for type in types:
    correct = sum([output['correct'] == True for output in full_outputs if output['question_type'] == type])
    total = sum([output['question_type'] == type for output in full_outputs])
    print(f"Type: {type}, Correct: {correct}, Total: {total}, Accuracy: {correct/total:.2%}")

print()
correct_full = sum([output['correct'] == True for output in full_outputs])
total_full = len(full_outputs)
print(f"Correct Full: {correct_full}, Total Full: {total_full}, Accuracy: {correct_full / total_full:.2%}")
