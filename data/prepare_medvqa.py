from datasets import load_dataset
from PIL import Image
import matplotlib.pyplot as plt
from huggingface_hub import hf_hub_download
from sentence_transformers import SentenceTransformer
import torch
from PIL import Image
from einops import repeat
from accelerate import Accelerator
from datasets import load_dataset
import sys
import os
import scipy
import pandas as pd
import numpy as np
import openai
import json
import requests
from PIL import Image

from io import BytesIO
import numpy as np
import pandas as pd
from datasets import DatasetDict
from itertools import islice

vqa_rad = load_dataset('flaviagiammarino/vqa-rad')

dataset = []
for i in vqa_rad['test']:
    dataset.append(i)

final_data = []
save_dir = "/data/MedXpertQA/images/"
datasetname = 'vqarad'
for idx, item in enumerate(dataset):
    label = item['answer'].lower()
    if label == 'yes' or label == 'no':
        id = datasetname + str(idx).zfill(5)
        question = item['question'] + "\nAnswer Choices: (A) yes (B) no"
        imgpath = []
        save_path = os.path.join(save_dir, id+".jpeg")
        item['image'].save(save_path, format="JPEG")
        imgpath.append({"image_path": id+".jpeg"})
        if label == 'yes':
            label = ["A"]
        else:
            label = ["B"]
        final_data.append({"id": id, "question": question, "images": imgpath, "label": label})

output_path = "/data/MedXpertQA/data/vqarad/input/vqarad_mm_input.jsonl"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
with open(output_path, 'w') as f:
    for item in final_data:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")


