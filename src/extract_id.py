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
torch.set_grad_enabled(False)
print("Disabled automatic differentiation")
import skdim

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

def convert_to_tensor(hs):
    return torch.stack([item for item in hs])

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
              "hidden_states": convert_to_tensor(hidden_states),
              "hidden_distances" : convert_to_distances(hidden_states),\
              "loss": loss.to(torch.float32).cpu().detach().numpy(), \
              # "logit_distances": torch.cdist(outputs.logits, outputs.logits).cpu().detach().numpy().squeeze()
             }         
      return ans

def get_knn_from_torch_distance_matrix(dist_matrix: torch.Tensor, k: int):
    """
    dist_matrix: torch.Tensor of shape (N, N) — pairwise distances
    k: number of nearest neighbors to extract (excluding self-distance)

    Returns:
        knn_dists: np.ndarray of shape (N, k) with distances to k nearest neighbors
    """
    assert dist_matrix.shape[0] == dist_matrix.shape[1], "Distance matrix must be square"
    
    # Exclude the self-distance (0) by sorting and skipping the first column
    sorted_dists, sorted_neighbors = torch.sort(dist_matrix, dim=1)
    
    return sorted_dists[:, 1:k+1], sorted_neighbors[:, 1:k+1]  # Only this small part is moved to CPU

def compute_id_across_estimators(rep, dist_matrix, method):
    if method == 'ESS':
        knn_dists, knn_neighbors = get_knn_from_torch_distance_matrix(dist_matrix, k=10)
        knn_dists, knn_neighbors = knn_dists.double().cpu().numpy(), knn_neighbors.cpu().numpy()
        ess=skdim.id.ESS()
        return ess.fit_transform(rep.cpu().numpy(), precomputed_knn_arrays=[knn_dists, knn_neighbors])
        
    elif method == 'GRIDE':
        _data = data.Data(distances=dist_matrix.cpu().numpy(), maxk=100)
        return _data.return_id_scaling_gride(range_max=64)[0][1]
        
    elif method == 'TLE':
        knn_dists, knn_neighbors = get_knn_from_torch_distance_matrix(dist_matrix, k=20)
        knn_dists, knn_neighbors = knn_dists.double().cpu().numpy(), knn_neighbors.cpu().numpy()
        tle=skdim.id.TLE()
        return tle.fit_transform(rep.cpu().numpy(), precomputed_knn_arrays=[knn_dists, knn_neighbors])
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
def compute_ids(all_layer_activations, all_layer_distances):
    ids = []
    for layer_activations, layer_distances in zip(all_layer_activations, all_layer_distances):
        ids_across_estimators = {}
        for method in ['GRIDE', 'ESS', "TLE"]:
            ids_across_estimators[method] = compute_id_across_estimators(layer_activations, layer_distances, method)
        ids.append(ids_across_estimators)
    return ids

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
    
    ds = load_dataset("NeelNanda/pile-10k")['train'] # type: ignore
    sequences = ds['text'] # type: ignore
    max_length = 1024
    
    output_folder = f"{args.input_dir}/Pile-{args.method.capitalize()}/{args.model_name}"
    os.makedirs(output_folder, exist_ok=True)
    if args.method == "structured":
        batch_sz = 32
        filtered_indices = np.load('filtered_indices.npy')
        filtered_sequences = [sequences[idx] for idx in filtered_indices]
        ids_output, losses = [], []
        
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
        
        np.save(f'{output_folder}/losses.npy', losses)
        np.save(f'{output_folder}/gride.npy', np.array(ids_output))
    
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
