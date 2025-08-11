import json
import os
with open("/data/MedXpertQA/data/usmlestep3/step3.json", "r", encoding="utf-8") as f:
    data = json.load(f)

with open("/data/MedXpertQA/data/usmlestep3/step3_solutions.json", "r", encoding="utf-8") as f:
    ans = json.load(f)

final_data = []


datasetname = 'step3-'
for idx, item in enumerate(data):

    id = datasetname + str(item['no']).zfill(5)
    question = item['question'] + "\nAnswer Choices: "
    options = []
    for itm in item['options'].items():
        question += " ("+itm[0]+") "+itm[1]+" "
        options.append({
            "letter": itm[0],
            "content": itm[1]
        })

    label = [ans[str(item['no'])]]
    final_data.append({"id": id, "question": question, "options": options, "label": label})

output_path = "/data/MedXpertQA/data/usmlestep3/input/usmlestep3_text_input.jsonl"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
with open(output_path, 'w') as f:
    for item in final_data:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")
