import json
import os

jsonfile = []
with open('/data/MedXpertQA/data/medqataiwan/test.jsonl','r') as f:
    for line in f:
        if line.strip(): 
            jsonfile.append(json.loads(line))
    
final_data = []
idf = 'test-'
for index, item in enumerate(jsonfile):
    id = idf + str(index).zfill(5)
    question = item['question'] + "\n答案選項: (A) " + item['options']['A'] + " (B) " + item['options']['B'] + " (C) " + item['options']['C'] + " (D) " + item['options']['D']
    label = [item['answer_idx']]

    final_data.append({"id": id, "question": question, "label": label})

output_path = "/data/MedXpertQA/data/medqataiwan/input/medqataiwan_text_input.jsonl"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
with open(output_path, 'w') as f:
    for item in final_data:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")