import os
import json
from model import APIAgent

# Constants
METHOD_LIST = ["zero_shot", "few_shot"]
DEMO_TYPE_LIST = ["ao", "cot"]

CONFIG_PATH = "config"
INPUT_PATH = "data/{dataset}/input/"
DEMO_PATH = "data/{dataset}/demonstrations/"

# Load configurations once
def load_config(file_path):
    with open(file_path, "r") as f:
        return json.load(f)

# Load model information and initialize the model if valid
def load_model(model, model_info):
    API_MODEL = model_info.get("API_MODEL", [])
    if model in API_MODEL:
        return APIAgent(model)
    else:
        raise ValueError(f"Unknown model: {model}")

# Setup function refactored to optimize config loading and checks
def setup(model, dataset, method, demo_type):
    # Load all config files once
    model_info = load_config(os.path.join(CONFIG_PATH, "model_info.json"))
    dataset_info = load_config(os.path.join(CONFIG_PATH, "dataset_info.json"))

    # Validate the inputs
    if dataset not in dataset_info:
        raise ValueError(f"Undefined dataset {dataset} in dataset_info.json")
    if method not in METHOD_LIST:
        raise ValueError(f"Undefined method {method} in METHOD_LIST")
    if demo_type not in DEMO_TYPE_LIST:
        raise ValueError(f"Undefined demo type {demo_type} in DEMO_TYPE_LIST")

    # Load the model and tasks
    model = load_model(model, model_info)
    tasks = dataset_info[dataset]["tasks"]

    return model, tasks
