import numpy as np
from tqdm import tqdm
from dadapy import data
from joblib import Parallel, delayed
from utils import parse_arguments

def compute_ids_for_all_layers(reps):
    all_ids = []
    # Iterating through the layers
    for full_rep in reps[1:]: # Not calculating ID for embedding layer
        _, indices = np.unique(full_rep, axis = 0, return_index=True)
        # Removing duplicate tokens
        rep = full_rep[indices, :][:, indices] 
        _data = data.Data(distances = rep, maxk = 300)
        # Calculating GRIDE up to range scaling = 256
        all_ids.append(_data.return_id_scaling_gride(range_max = 256)) 
    return np.array(all_ids)

def compute_ids(input_file):
    """
    Computes the layerwse IDs for a batch of prompts.

    Parameters
    ----------
    input_file : string
        Path to the .npy file that stores the layerwise distance matrices for 
        a batch of prompts.

    Returns
    -------
    numpy array
        The numpy array containing the layerwise ID values for the 
        batch of sequences.   

    """
    reps = np.load(input_file)
    all_ids = []
    for rep in tqdm(reps): # 
        all_ids.append(compute_ids_for_all_layers(rep))
    return np.array(all_ids)

def read_and_write_gride(input_dir, n_batches):
    ids = Parallel(n_jobs=-1)(delayed(compute_ids)(f"{input_dir}/distances/{idx}.npy")\
                              for idx in range(n_batches)) 
    np.save(f"{input_dir}/summaries/gride.npy", np.concatenate(ids, axis = 0))


def compute_csn(input_file):
    reps = np.load(input_file)
    all_csn = []
    for rep in tqdm(reps): # Iterating through prompts in this batch
        all_csn.append([np.mean(rep, axis = (-1, -2)), np.std(rep, axis = (-1, -2))])
    return np.array(all_csn)

def read_and_write_csn(input_dir, n_batches):
    csn = Parallel(n_jobs=-1)(delayed(compute_csn)(f"{input_dir}/cosine_similarities/{idx}.npy")\
                              for idx in range(n_batches)) 
    np.save(f"{input_dir}/summaries/csn_sim.npy", np.concatenate(csn, axis = 0))
    
def compute_neighborhood_overlap(rep, next_rep, maxk = 32):
    rep = np.argsort(rep, axis = 1)
    next_rep = np.argsort(next_rep, axis = 1)
    
    n = len(rep)
    # Initialize the answer array to hold the number of overlaps at each knn level
    ans = np.zeros((n, maxk))
    
    for idx in range(n): # Iterating through tokens
        
        # For each index, create a boolean array for quick lookups
        rep_flags = np.zeros(np.max(rep) + 1, dtype=bool)
        next_rep_flags = np.zeros(np.max(next_rep) + 1, dtype=bool)
        
        # Iterate over knn and count the number of intersections
        for knn in range(1, maxk):
            rep_flags[rep[idx, knn]] = True
            next_rep_flags[next_rep[idx, knn]] = True
            ans[idx, knn] = np.sum(rep_flags & next_rep_flags)/knn 
    
    return ans.mean(axis=0)

def compute_nos_for_all_layers(reps):
    all_nos = []
    for rep, next_rep in zip(reps[1:-1], reps[2:]): # Not calculating NO for embedding layer
        all_nos.append(compute_neighborhood_overlap(rep, next_rep))
    return np.array(all_nos)

def compute_nos(input_file):
    reps = np.load(input_file)
    all_nos = []
    for rep in tqdm(reps):
        all_nos.append(compute_nos_for_all_layers(rep))
    return np.array(all_nos)

def read_and_write_no(input_dir, n_batches):
    mnos = Parallel(n_jobs=-1)(delayed(compute_nos)(f"{input_dir}/distances/{idx}.npy")\
                              for idx in range(n_batches)) 
    np.save(f"{input_dir}/summaries/mnos.npy", np.concatenate(mnos, axis = 0))
    
if __name__ == "__main__":
    """
        This code takes in model, method(structured or shuffled) as the input
        and stors the intrinsic dimension (using GRIDE), 
        cosine similarities and the neighborhood overlap using the 
        distance matrix and cosine similarities calculated using extract.py. 
    """
    args = parse_arguments()
    
    input_dir = args.input_dir
    model_name = args.model_name
    
    method = args.method.lower()
    if method not in ["shuffled", "structured"]:
        raise ValueError("Check args.method. It should be either structured or shuffled.")
    
    if method == 'structured':
        dataset_name = 'Pile-Structured'
        n_batches = 71
    elif method == 'shuffled':
        dataset_name = 'Pile-Shuffled'
        n_batches = 50
    
    input_dir = f"{args.input_dir}/{dataset_name}/{model_name}"
    read_and_write_gride(input_dir, n_batches = n_batches)
    read_and_write_csn(input_dir, n_batches = n_batches)
    read_and_write_no(input_dir, n_batches = n_batches)
