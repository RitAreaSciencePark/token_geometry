import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from huggingface_hub import login
from tqdm import tqdm
from joblib import Parallel, delayed
from utils import extract_hidden_states, parse_arguments, \
    compute_cosine_similarities, compute_distances, shuffle_tokens, \
        convert_to_numpy, compute_nearest_neighbors, compute_nn_sim
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

if __name__ == "__main__":
    """
        This code takes in model, method(structured or shuffled) as the input
        and stores the distance matrices, losses and cosine similarites in 
        batches. 
    """
    
    args = parse_arguments()
    
    # Login to Hugging Face Hub with the provided token for authentication 
    login_token = args.login_token
    login(token=login_token)

    # Dictionary mapping model names to their paths on Hugging Face Hub
    path_dict ={"Llama-3-8B": "meta-llama/Meta-Llama-3-8B", \
               "Mistral-7B" :"mistralai/Mistral-7B-v0.1", \
            "Pythia-6.9B"  :"EleutherAI/pythia-6.9b",\
        "Pythia-6.9B-Deduped"  :"EleutherAI/pythia-6.9b-deduped"}
    model_name = args.model_name
    if model_name not in path_dict.keys():
        raise ValueError(f"{model_name} not supported yet. The model names \
                         currently supported are {path_dict.keys()}")
    path = path_dict[model_name]
    # Load the tokenizer and model from the pre-trained model on Hugging Face
    tokenizer = AutoTokenizer.from_pretrained(path)   
    model = AutoModelForCausalLM.from_pretrained(path,device_map="auto", \
                                        torch_dtype=torch.bfloat16)
    
    input_dir = args.input_dir
    method = args.method.lower()
    if method not in ["shuffled", "structured"]:
        raise ValueError("Check args.method. It should be either structured or shuffled.")
    
    print(f"Model = {model_name}, path = {path}, method = {method}")
    
    # Load the Pile-10K dataset from the Hugging Face datasets library
    ds = load_dataset("NeelNanda/pile-10k")['train']
    sequences = ds['text']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    max_length = 1024 # Maximum sequence length for the model
    
    if method == "structured":
        
        # Batch size for processing. This is also the number of prompts to be 
        # saved in each .npy file for processing
        batch_sz = 32
        # Load precomputed filtered indices. This contains the indices of the 
        # prompts with atleast 1024 tokens according to the tokenization schemes 
        # of Llama-3-8B, Mistral-7B-v0.1 and Pythia-6.9B
        filtered_indices = np.load('filtered_indices.npy')
        # Storing the corresponding prompts
        filtered_sequences = [sequences[idx] for idx in filtered_indices]
        
        # Folder to save the distance matrices, cosine similarities and losses.
        # Making sure the subfolders exist and create them if necessary.
        output_folder = f"{input_dir}/Pile-Structured/{model_name}"
        os.makedirs(f"{output_folder}/distances", exist_ok=True)
        os.makedirs(f"{output_folder}/cosine_similarities", exist_ok=True)
        os.makedirs(f"{output_folder}/losses", exist_ok=True)
        
        # Iterate over batches of filtered sequences    
        for batch_start in tqdm(range(0, len(filtered_indices), batch_sz)):
            batch_sequences = filtered_sequences[batch_start: batch_start + batch_sz]
            intermediate_reps = [extract_hidden_states(sequence, model, \
                        tokenizer, max_length = max_length) \
                                 for sequence in batch_sequences]
            hidden_states = np.array([item["hidden_states"] for item in intermediate_reps])
            losses = np.array([item["loss"] for item in intermediate_reps])
            distances = Parallel(n_jobs=-1)(delayed(compute_distances)(hs) for hs in hidden_states)
            csn_sim = Parallel(n_jobs=-1)(delayed(compute_cosine_similarities)(hs) for hs in hidden_states)
            batch_num = batch_start//batch_sz
            np.save(f"{output_folder}/distances/{batch_num}.npy", distances)
            np.save(f"{output_folder}/cosine_similarities/{batch_num}.npy", csn_sim)
            np.save(f"{output_folder}/losses/{batch_num}.npy", losses)
    
    # Processing for 'shuffled' experiment
    elif method == "shuffled":
        batch_sz = 1
        # =============================================================================
        # Here is the code to regenerate the subset_indices.npy
        # np.random.seed(42)
        # lengths = np.array([len(tokenizer(sequence.strip())['input_ids']) \
        #    for sequence in tqdm(sequences)])
        # indices =  np.where(lengths >= max_length)[0]
        # new_filtered_indices = np.array(sorted(np.random.choice(indices, 50, replace = False)))
        # np.save(f"{output_folder}/subset_indices.npy", new_filtered_indices)
        # =============================================================================
        
        new_filtered_indices = np.load('subset_indices.npy')
        filtered_sequences = [sequences[idx] for idx in new_filtered_indices]
        
        output_folder = f"{input_dir}/Pile-Shuffled/{model_name}"
        os.makedirs(f"{output_folder}/distances", exist_ok=True)
        os.makedirs(f"{output_folder}/cosine_similarities", exist_ok=True)
        os.makedirs(f"{output_folder}/summaries", exist_ok=True)
        losses = []
        all_nn_sims =[]
        for batch_num, test_seq in enumerate(tqdm(filtered_sequences)):
            with torch.no_grad():
                hidden_states = []
                for idx in range(6):
                    inputs = tokenizer(test_seq.strip(), add_special_tokens = False, \
                                       return_tensors = "pt", max_length = max_length, \
                                                 truncation=True).to(device)
                    ids = inputs['input_ids']
                    new_ids = shuffle_tokens(ids, idx).to(device)
                    inputs = {'input_ids':new_ids}
                    outputs = model(**inputs, labels = new_ids.clone(), output_hidden_states=True)
                    hidden_state, loss = outputs.hidden_states, outputs.loss
                    hidden_states.append(convert_to_numpy(hidden_state))
                    losses.append(loss.to(torch.float32).cpu().detach().numpy())
            
            distances = Parallel(n_jobs=-1)(delayed(compute_distances)(hs) \
                        for hs in hidden_states) # iterating through prompts
            if args.find_nn_sim == "True":
                nearest_nbrs = np.array(Parallel(n_jobs=-1)(delayed(compute_nearest_neighbors)(dm)\
                                                   for dm in distances))
                # nearest_nbrs.shape = 6 x 33 x 1024 x 2
                nn_sim = Parallel(n_jobs=-1)(delayed(compute_nn_sim)(hs, nearest_nbr_list)\
                            for hs, nearest_nbr_list in zip(hidden_states, nearest_nbrs))
                all_nn_sims.append(nn_sim)
                
            csn_sim = Parallel(n_jobs=-1)(delayed(compute_cosine_similarities)(hs) for hs in hidden_states)        
            np.save(f"{output_folder}/distances/{batch_num}.npy", distances)
            np.save(f"{output_folder}/cosine_similarities/{batch_num}.npy", csn_sim)
        np.save(f"{output_folder}/summaries/losses.npy", losses)
        if args.find_nn_sim == "True": 
            np.save(f"{output_folder}/summaries/nn_sim.npy", all_nn_sims) 