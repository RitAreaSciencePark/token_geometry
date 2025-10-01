import numpy as np
import os
from tqdm import tqdm
from joblib import Parallel, delayed
from dadapy import Data
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse

import argparse
import json
import matplotlib.pyplot as plt
import skdim
import multiprocessing
import faiss



def get_data(benign=True,path = "tmp_path"):
    filenames = os.listdir(path)
    
    start_word = 'benign'
    if not benign:
        start_word = 'jailbreak'
    
    #takes the file that start with benign or jailbreak
    files =  [x for x in filenames if start_word in x]
    filepaths = [path+os.sep+f for f in files]

    return filepaths

def compute_dimension(input_data,method="GRIDE"):

    if method == "GRIDE":
        data = Data(coordinates=input_data)
        ids_scaling, ids_scaling_err, rs_scaling = data.return_id_scaling_gride(range_max = 32)
        return ids_scaling
    elif method == "MADA":
        # CORRECTED: Instantiate and call in one line
        return skdim.id.MADA().fit_transform(input_data)
    
    elif method == "lPCA":
        # CORRECTED: Instantiate and call in one line
        return skdim.id.lPCA().fit_transform(input_data)

    elif method == "ESS":
        # CORRECTED: Instantiate and call in one line
        return skdim.id.ESS().fit_transform(input_data,n_neighbors=10)
    elif method == "CorrInt":
        return skdim.id.CorrInt().fit_transform(input_data)
    elif method == "MLE":
        n_samples = input_data.shape[0]
        # MLE requires at least 2 points to compute neighbors.
        if n_samples < 2:
            print(f"[WARNING] - Skipping MLE computation: not enough samples ({n_samples}).")
            return None
        
        # Set k to be less than the number of samples.
        # Default is 20, we'll use that as a max.
        k = min(4, n_samples - 1)
        
        try:
            # Pass the dynamically calculated 'k' to the estimator.
            return skdim.id.MLE(K=k).fit_transform(input_data,n_jobs=1)
        except Exception as e:
            print(f"[ERROR] MLE method failed with k={k} and n_samples={n_samples}: {e}")
            return None
    elif method == "TLE":
        # ADDED: try-except block to prevent crashes from the TLE method
        try:
            # Attempt to compute the dimension using TLE
            return skdim.id.TLE().fit_transform(input_data)
        except Exception as e:
            # If any error occurs, print it and return None
            print(f"[ERROR] TLE method failed with an exception: {e}")
            return None

    elif method == "MOM":
        return skdim.id.MOM().fit_transform(input_data)
    else:
        print("[ERROR]: Method not recognized :( ")
    return None


def read_files_and_run_layer(input_files, output_path, output_file, layer_to_compute, method="GRIDE", batch_size=25):
    """
    Computes the dimension for a single, specified layer and saves the results.

    Args:
        input_files (list): A list of paths to the input .npy files.
        output_path (str): The path to the output directory.
        output_file (str): The base name for the output file.
        layer_to_compute (int): The index of the layer to compute.
        method (str, optional): The method for dimension computation. Defaults to "GRIDE".
    """
    try:
        test_file = np.load(input_files[0], allow_pickle=True)
        N_LAYERS = test_file[0].shape[0]
        print(f"[INFO] - Total layers available in files: {N_LAYERS}")
    except Exception as e:
        print(f"[ERROR] - Could not read shape from {input_files[0]}: {e}")
        return

    # --- Validate the requested layer index ---
    if not (0 <= layer_to_compute < N_LAYERS):
        print(f"[ERROR] - Invalid layer index '{layer_to_compute}'. Must be between 0 and {N_LAYERS - 1}.")
        return

    # --- MODIFICATION: Create a specific directory for the layer's output ---
    layer_output_path = os.path.join(output_path, f"layer_{layer_to_compute}")
    os.makedirs(layer_output_path, exist_ok=True)
    print(f"[INFO] - Output will be saved in: {layer_output_path}")

    print(f"[INFO] - Starting computation for layer {layer_to_compute} with method '{method}'")

    # --- MODIFICATION: Initialize batch processing variables ---
    batch_dimensions = []
    batch_counter = 0
    total_results_count = 0

    # --- Process all files for the specified layer ---
    for f in tqdm(input_files, desc=f"Processing Files for Layer {layer_to_compute}"):
        try:
            tmp_data = np.load(f, allow_pickle=True)
            N_PROMPTS = tmp_data.shape[0]
            
            for p in range(N_PROMPTS):
                # Directly access the data for the specified layer
                input_matrix = tmp_data[p][layer_to_compute]
                
                resulting_dimension = compute_dimension(input_data=input_matrix, method=method)
                
                if resulting_dimension is not None and resulting_dimension.size > 0:
                    batch_dimensions.append(resulting_dimension)
                    total_results_count += 1

                    # --- MODIFICATION: Check if the batch is full and save it ---
                    if len(batch_dimensions) == batch_size:
                        batch_filename = f"{method}_{output_file}_batch_{batch_counter}.npy"
                        batch_output_path = os.path.join(layer_output_path, batch_filename)
                        
                        np.save(batch_output_path, np.array(batch_dimensions))
                        
                        # Reset for the next batch
                        batch_dimensions = []
                        batch_counter += 1

        except Exception as e:
            print(f"[WARNING] - Could not process file {f}: {e}")
            continue

    # --- MODIFICATION: Save any remaining results in the last batch ---
    if batch_dimensions:
        batch_filename = f"{method}_{output_file}_batch_{batch_counter}.npy"
        batch_output_path = os.path.join(layer_output_path, batch_filename)
        
        np.save(batch_output_path, np.array(batch_dimensions))
        print(f"[INFO] - Saved final batch {batch_counter} to {batch_output_path}")
        batch_counter += 1

    print(f"[INFO] - Finished computation.")
    print(f"[INFO] - Found and saved {total_results_count} valid results in {batch_counter} batch files.")


if __name__ == "__main__":
    benign_filepath = get_data(path="tmp_path")
    jailbreaks_filepath = get_data(benign=False,path="tmp_path")
    layer_to_compute = 5    

    read_files_and_run_layer(
        input_files=jailbreaks_filepath,
        output_path="<output path>",
        output_file="jailbreaks",
        layer_to_compute=layer_to_compute,
        method="GRIDE"
    )
