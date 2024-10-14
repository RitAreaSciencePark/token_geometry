import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from huggingface_hub import login
from tqdm import tqdm
from joblib import Parallel, delayed
from utils import get_path, parse_arguments, convert_to_numpy, compute_distances, \
    compute_cosine_similarities, shuffle_tokens
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

args = parse_arguments()
login_token = args.login_token
login(token=login_token)

model_name = args.model_name
path = get_path(model_name)
tokenizer = AutoTokenizer.from_pretrained(path)   
model = AutoModelForCausalLM.from_pretrained(path,device_map="auto", torch_dtype=torch.bfloat16)

input_dir = args.input_dir
print(f"Model = {model_name}, path = {path}")

ds = load_dataset("NeelNanda/pile-10k")['train']
