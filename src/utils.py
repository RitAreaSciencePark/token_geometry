import numpy as np
import torch
import argparse
import json
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from scipy.spatial import distance

def parse_arguments():
    parser = argparse.ArgumentParser()   
    parser.add_argument("--input_dir", type=str, default=None)
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--method", type=str, default=None)
    parser.add_argument("--login_token", type=str, default=None)
    parser.add_argument("--find_nn_sim", type=str, default="False")
    args = parser.parse_args()
    print("input args:\n", json.dumps(vars(args), indent=4, separators=(",", ":")))
    return args

def convert_to_numpy(hs):
    return np.array([item.to(torch.float32).cpu().detach().numpy() for item in hs])

def extract_hidden_states(sequence, model, tokenizer, max_length):
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  with torch.no_grad():  # Disable gradient computation  
      inputs = tokenizer(sequence.strip() , add_special_tokens = False, \
                         return_tensors = "pt", max_length = max_length, \
                             truncation=True).to(device)
      outputs = model(**inputs, labels = inputs['input_ids'].clone(), output_hidden_states=True)
      hidden_states, loss = outputs.hidden_states, outputs.loss
  return {"hidden_states" : convert_to_numpy(hidden_states),\
          "loss": loss.to(torch.float32).cpu().detach().numpy()}

def compute_distances(hs):
    # Iterating through layers
    return  np.array([euclidean_distances(embeddings[0]) for embeddings in hs])

def compute_nearest_neighbors(distances):
    return  np.array([np.argsort(distance, axis = -1)[:, 1:3]  for distance in distances])

def csn_sim(embeddings, nn_list):
    ans = []
    for idx, (token_rep, nn) in enumerate(zip(embeddings, nn_list)):
        v1 = embeddings[idx] - embeddings[nn_list[idx, 0]]
        v2 = embeddings[idx] - embeddings[nn_list[idx, 1]]
        ans.append(1 - distance.cosine(v1, v2))
    return np.array(ans)

def compute_nn_sim(hs, nn_lists):
    return  np.array([csn_sim(embeddings[0], nn_list)  for embeddings, nn_list in zip(hs, nn_lists)])

def compute_cosine_similarities(hs):
    return  np.array([cosine_similarity(embeddings[0]) for embeddings in hs])

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