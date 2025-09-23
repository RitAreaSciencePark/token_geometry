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

def shuffle_tokens(ids, shuffle_index):
    """
    For the shuffle experiment described in Algorithm 1 in the paper - 
    'The Geometry of Tokens in Internal Representations of Large Language Models'

    Parameters
    ----------
    ids : torch.tensor with dtype integer
        input_ids of the tokens for a single prompt that needs to be shuffled.
        
    shuffle_index : integer between 0 and 6 (not including 6).
        the degree of shuffling where 0 is no shuffle and 5 is fully shuffled
        case        

    Returns
    -------
    new_ids : torch.tensor with dtype integer
        the shuffled ids for the given prompt.

    """
    N, K = ids.shape[-1], 4**shuffle_index
    block_size = N//K
    permutation = np.random.permutation(K)
    new_ids = ids.reshape((1, K, block_size))
    new_ids = new_ids[0, permutation, :]
    new_ids = new_ids.reshape(1, N)
    return new_ids

def parse_arguments():
    parser = argparse.ArgumentParser()   
    parser.add_argument("--input_dir", type=str, default=None)
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--method", type=str, default=None)
    parser.add_argument("--batch_start", type=int, required=True, help="Start index of batch")
    parser.add_argument("--batch_end", type=int, required=True, help="End index of batch (exclusive)")
    
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
        dist_matrix_np = dist_matrix.cpu().numpy()
        _, indices = np.unique(dist_matrix_np, axis=0, return_index=True)
        unique_dist_matrix = dist_matrix_np[indices, :][:, indices]
        _data = data.Data(distances=unique_dist_matrix, maxk=100)
        return np.array(_data.return_id_scaling_gride(range_max=64))
        
    elif method == 'TLE':
        knn_dists, knn_neighbors = get_knn_from_torch_distance_matrix(dist_matrix, k=20)
        knn_dists, knn_neighbors = knn_dists.double().cpu().numpy(), knn_neighbors.cpu().numpy()
        tle=skdim.id.TLE()
        return tle.fit_transform(rep.cpu().numpy(), precomputed_knn_arrays=[knn_dists, knn_neighbors])
    
    else:
        raise ValueError(f"Unknown method: {method}")

METHODS = ['GRIDE', 'ESS', 'TLE']
  
def compute_ids(all_layer_activations, all_layer_distances):
    # import ipdb; ipdb.set_trace()
    assert all_layer_activations.shape[:-1] == all_layer_distances.shape[:-1]
    ids_across_estimators = {}
    
    for method in METHODS: 
        ids_across_estimators[method] = [] 
    
    for layer_activations, layer_distances in zip(all_layer_activations, all_layer_distances):
        for method in METHODS:
            ids_across_estimators[method].append(compute_id_across_estimators(layer_activations, layer_distances, method))
    
    for method in METHODS:
        ids_across_estimators[method] = np.array(ids_across_estimators[method])
        if method == 'GRIDE':
            assert ids_across_estimators[method].shape == (len(all_layer_activations), 3, 6), f"Shapes don't match, {ids_across_estimators[method].shape}, {(len(all_layer_activations), 3, 6)}"
        elif method == 'TLE' or method == 'ESS':
            expected_shape = (len(all_layer_activations),)
            assert ids_across_estimators[method].shape == expected_shape, \
                f"Shapes don't match, got {ids_across_estimators[method].shape}, expected {expected_shape}"
        else:
            raise ValueError(f"Unknown method: {method}")
        
    return ids_across_estimators

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
    model.eval()
    ds = load_dataset("NeelNanda/pile-10k")['train'] # type: ignore
    sequences = ds['text'] # type: ignore
    max_length = 1024
    
    output_folder = f"{args.input_dir}/Pile-{args.method.capitalize()}/{args.model_name}"
    os.makedirs(output_folder, exist_ok=True)
    if args.method == "structured":
        filtered_indices = np.load('filtered_indices.npy')[args.batch_start:args.batch_end]
        filtered_sequences = [sequences[idx] for idx in filtered_indices]
        ids_output, losses = [], []
        result = {
                "ESS": [],
                "TLE": [],
                "GRIDE": [],
                "loss": []
            }
        for sequence in tqdm(filtered_sequences):
            intermediate_reps = [
                extract_hidden_states(sequence, model, tokenizer, max_length=max_length)
            ]
            hidden_states = [item["hidden_states"][1:, 0] for item in intermediate_reps]
            hidden_distances = [item["hidden_distances"][1:] for item in intermediate_reps]
            # hidden_ids = np.array(Parallel(n_jobs=-1)(delayed(compute_ids)(hs) for hs in hidden_distances))
            # hidden_ids = Parallel(n_jobs=2, verbose=1)(delayed(compute_ids)(hs, hd) for hs, hd in zip(hidden_states, hidden_distances))
            hidden_ids = [compute_ids(hs, hd) for hs, hd in zip(hidden_states, hidden_distances)]
            
            assert len(hidden_ids) == 1
            sequence_ids = hidden_ids[0]
            for method in METHODS: result[method].append(sequence_ids[method])
            
            assert len(intermediate_reps) == 1
            result["loss"].append(intermediate_reps[0]["loss"])
            torch.cuda.empty_cache()
            # import ipdb; ipdb.set_trace()
        
        # np.save(f'{output_folder}/losses.npy', losses)
        # np.save(f'{output_folder}/gride.npy', np.array(ids_output))
        output_file = f"{output_folder}/results_{args.batch_start}_{args.batch_end}.npz"
        np.savez_compressed(output_file, **result)
        print(f"✅ File saved to {output_file}")
        
    elif args.method == "shuffled":
        filtered_indices = np.load('subset_indices.npy')[args.batch_start:args.batch_end]
        filtered_sequences = [sequences[idx] for idx in filtered_indices]
        ids_output, losses = [], []
        result = {
                "ESS": [],
                "TLE": [],
                "GRIDE": [],
                "loss": []
            }
        for sequence in filtered_sequences:
            inputs = tokenizer(sequence.strip(), add_special_tokens = False, return_tensors = "pt", 
                               max_length = max_length, truncation=True).to(device)
            for shuffle_idx in tqdm(range(6)):
                ids = inputs['input_ids'].clone()
                new_ids = shuffle_tokens(ids, shuffle_idx).to(device)
                inputs = {'input_ids':new_ids}
                outputs = model(**inputs, labels = new_ids.clone(), output_hidden_states=True)
                hidden_states, loss = outputs.hidden_states, outputs.loss
                ans = {
                        "hidden_states": convert_to_tensor(hidden_states),
                        "hidden_distances" : convert_to_distances(hidden_states),\
                        "loss": loss.to(torch.float32).cpu().detach().numpy(), \
                        # "logit_distances": torch.cdist(outputs.logits, outputs.logits).cpu().detach().numpy().squeeze()
                        }   
                
                intermediate_reps = [ans]
                hidden_states = [item["hidden_states"][1:, 0] for item in intermediate_reps]
                hidden_distances = [item["hidden_distances"][1:] for item in intermediate_reps]
                # hidden_ids = np.array(Parallel(n_jobs=-1)(delayed(compute_ids)(hs) for hs in hidden_distances))
                # hidden_ids = Parallel(n_jobs=2, verbose=1)(delayed(compute_ids)(hs, hd) for hs, hd in zip(hidden_states, hidden_distances))
                hidden_ids = [compute_ids(hs, hd) for hs, hd in zip(hidden_states, hidden_distances)]
                
                assert len(hidden_ids) == 1
                sequence_ids = hidden_ids[0]
                for method in METHODS: result[method].append(sequence_ids[method])
                
                assert len(intermediate_reps) == 1
                result["loss"].append(intermediate_reps[0]["loss"])
                torch.cuda.empty_cache()
            # import ipdb; ipdb.set_trace()
        
        # np.save(f'{output_folder}/losses.npy', losses)
        # np.save(f'{output_folder}/gride.npy', np.array(ids_output))
        output_file = f"{output_folder}/results_{args.batch_start}_{args.batch_end}.npz"
        np.savez_compressed(output_file, **result)
        print(f"✅ File saved to {output_file}")
