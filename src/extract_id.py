import torch
import numpy as np
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from tqdm import tqdm
from joblib import Parallel, delayed
from dadapy import data
import argparse
import json


os.environ["TOKENIZERS_PARALLELISM"] = "false"


def shuffle_tokens(ids):
    assert ids.shape[0] == 1 and len(ids.shape) == 2, f"Expected shape (1, N), but got {ids.shape}"
    permutation = np.random.permutation(ids.shape[1])
    return ids[:, permutation]

def parse_arguments():
    parser = argparse.ArgumentParser()   
    parser.add_argument("--input_dir", type=str, default=None)
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--method", type=str, default=None)
    args = parser.parse_args()
    print("input args:\n", json.dumps(vars(args), indent=4, separators=(",", ":")))
    return args

def convert_to_distances(hs):
    return torch.stack([torch.cdist(item, item).squeeze() for item in hs])

def extract_hidden_states(sequence, model, tokenizer, max_length):
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  with torch.no_grad():  # Disable gradient computation  
      inputs = tokenizer(sequence.strip() , add_special_tokens = False, \
                         return_tensors = "pt", max_length = max_length, \
                             truncation=True).to(device)
      outputs = model(**inputs, labels = inputs['input_ids'].clone(), \
                      output_hidden_states=True)
      hidden_states, loss = outputs.hidden_states, outputs.loss
      ans = {
              "hidden_distances" : convert_to_distances(hidden_states).cpu().detach().numpy(),\
              "loss": loss.to(torch.float32).cpu().detach().numpy(), \
              "logit_distances": torch.cdist(outputs.logits, outputs.logits).cpu().detach().numpy().squeeze()
             }         
      return ans


def compute_ids(full_reps):
    ids = []
    for full_rep in full_reps:
        _, indices = np.unique(full_rep, axis=0, return_index=True)
        rep = full_rep[indices, :][:, indices]
        _data = data.Data(distances=rep, maxk=100)
        ids.append(_data.return_id_scaling_gride(range_max=64))
    return np.array(ids)

def load_model(model_name, device):
    try:
        # Attempt to load the model and tokenizer
        model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        print(f"Model '{model_name}' is available on Hugging Face.")
        return model, tokenizer
    except Exception as e:
        raise ValueError(f"Model '{model_name}' not found on Hugging Face. Error: {str(e)}")

if __name__ == "__main__":
    # =============================================================================
    #     model_list = [
    #         "meta-llama/Meta-Llama-3-8B",
    #         "mistralai/Mistral-7B-v0.1",
    #         "EleutherAI/pythia-6.9b-deduped",
    #         "EleutherAI/pythia-160m-deduped",
    #         "EleutherAI/pythia-410m-deduped",
    #         "EleutherAI/pythia-1.4b-deduped",
    #         "EleutherAI/pythia-2.8b-deduped",
    #         "facebook/opt-6.7b",
    #         "gpt2",
    #         "gpt2-large",
    #         "gpt2-xl"
    #         ]
    # =============================================================================
    args = parse_arguments()
    model_name = args.model_name
    
    device = torch.device('cuda')
    model, tokenizer = load_model(model_name, device = device)
    
    ds = load_dataset("NeelNanda/pile-10k")['train']
    sequences = ds['text']
    max_length = 1024
    
    output_folder = f"{args.input_dir}/Pile-{args.method.capitalize()}/{args.model_name}"
    os.makedirs(output_folder, exist_ok=True)
    if args.method == "structured":
        batch_sz = 32
        filtered_indices = np.load('filtered_indices.npy')
        filtered_sequences = [sequences[idx] for idx in filtered_indices]
        ids_output, logits_ids_output, losses = [], [], []
        
        for batch_start in tqdm(range(0, len(filtered_indices), batch_sz)):
            batch_sequences = filtered_sequences[batch_start: batch_start + batch_sz]
            intermediate_reps = [
                extract_hidden_states(seq, model, tokenizer, max_length=max_length)
                for seq in batch_sequences
            ]
            hidden_distances = np.array([item["hidden_distances"][1:] for item in intermediate_reps])
            hidden_ids = np.array(Parallel(n_jobs=-1)(delayed(compute_ids)(hs) for hs in hidden_distances))
            ids_output.extend(hidden_ids)
            losses.extend([item["loss"] for item in intermediate_reps])
            logit_distances = np.array([item["logit_distances"] for item in intermediate_reps])
            logit_ids = np.array(Parallel(n_jobs=-1)(delayed(compute_ids)([ld]) for ld in logit_distances))
            logits_ids_output.extend(logit_ids)
        
        np.save(f'{output_folder}/losses.npy', losses)
        np.save(f'{output_folder}/gride.npy', np.array(ids_output))
        np.save(f'{output_folder}/logits_id.npy', np.array(logits_ids_output).squeeze())        
    
    elif args.method == "shuffled":
        new_filtered_indices = np.load('subset_indices.npy')
        filtered_sequences = [sequences[idx] for idx in new_filtered_indices]
        all_losses, all_ids = [], []
        
        for test_seq in tqdm(filtered_sequences):
            intermediate_reps, losses = [], []
            for _ in range(20):
                with torch.no_grad():
                    inputs = tokenizer(test_seq.strip(), add_special_tokens=False, return_tensors="pt",
                                       max_length=max_length, truncation=True).to(device)
                    ids = inputs['input_ids']
                    new_ids = shuffle_tokens(ids).to(device)
                    inputs = {'input_ids':new_ids}
                    outputs = model(**inputs, labels = inputs['input_ids'].clone(), output_hidden_states=True)
                    hidden_states, loss = outputs.hidden_states, outputs.loss
                    intermediate_reps.append(convert_to_distances(outputs.hidden_states).cpu().detach().numpy())
                    losses.append(outputs.loss.to(torch.float32).cpu().detach().numpy())
            
            hidden_distances = np.array([item[1:] for item in intermediate_reps])
            hidden_ids = np.array(Parallel(n_jobs=-1)(delayed(compute_ids)(hs) for hs in hidden_distances))
            all_ids.extend(hidden_ids)
            all_losses.extend(losses)
        
        np.save(f"{output_folder}/losses.npy", all_losses)
        np.save(f"{output_folder}/gride.npy", all_ids)