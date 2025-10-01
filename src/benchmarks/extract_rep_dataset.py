import numpy as np
from transformers import AutoModel, AutoTokenizer
import torch
import os
import json
from tqdm import tqdm
import queue
import threading
from datasets import load_dataset, Dataset
from compute_guard_models import  get_dataset_malicious,get_dataset_jailbreak,get_dataset_attack

MODEL_BASE = "meta-llama/Meta-Llama-3-8B"

def save_worker(q, saved_chunk_paths):
    """
    A worker function that runs in a separate thread.
    It continuously fetches data from the queue and saves it to disk.
    """
    while True:
        try:
            # Get data from the queue. The 'get' method will block until an item is available.
            # A timeout is added to prevent it from blocking indefinitely if the main thread hangs.
            item = q.get(timeout=60)

            # The sentinel value 'None' signals the worker to terminate.
            if item is None:
                print("\n💾 Save worker received termination signal.")
                break

            chunk_array, chunk_file_path, chunk_number = item

            # Save the chunk to its own file
            np.save(chunk_file_path, chunk_array, allow_pickle=True)
            print(f"\n✅ Saved chunk {chunk_number} with {len(chunk_array)} items to: {chunk_file_path}")
            saved_chunk_paths.append(chunk_file_path)

            # Mark the task as done
            q.task_done()

        except queue.Empty:
            print("\n💾 Save worker timed out. Assuming processing is complete.")
            break
        except Exception as e:
            print(f"\n🚨 Error in save worker: {e}")
            break



def get_hidden_states_from_prompts_optimized(model, tokenizer, prompts, output_file_path, batch_size=4, chunk_size=50):
    """
    Gets hidden states from prompts and saves them in memory-efficient chunks using a parallel save worker.

    Args:
        model: A pre-trained transformer model.
        tokenizer: The tokenizer corresponding to the model.
        prompts (list): A list of strings (prompts).
        output_file_path (str): The base path for the output files.
        batch_size (int): The number of prompts to process in each batch on the GPU.
        chunk_size (int): The number of prompts to save in each file.

    Returns:
        list: A list of file paths for the saved chunks.
    """
    # A thread-safe queue to hold data chunks ready for saving.
    save_queue = queue.Queue(maxsize=4) # maxsize prevents the queue from using too much memory
    saved_chunk_paths = []

    # Start the save worker thread
    saver_thread = threading.Thread(target=save_worker, args=(save_queue, saved_chunk_paths))
    saver_thread.start()
    print("\n🚀 Save worker thread started.")

    current_chunk_states = []
    chunk_number = 1
    num_prompts = len(prompts)
    base_name, ext = os.path.splitext(output_file_path)

    print(f"\nProcessing {num_prompts} prompts in batches of {batch_size}...")

    # The main thread will focus on inference (the "producer")
    for i in tqdm(range(0, num_prompts, batch_size), desc="🚀 Processing Batches"):
        batch_prompts = prompts[i:i + batch_size]

        inputs = tokenizer(batch_prompts, return_tensors='pt', padding=True, truncation=True)
        # Move inputs to the same device as the model
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        # Move hidden states to CPU and convert to NumPy to free up GPU memory quickly
        all_layers_hidden_states = torch.stack(outputs.hidden_states).cpu()
        batch_hidden_states_per_prompt = np.transpose(all_layers_hidden_states.numpy(), (1, 0, 2, 3))

        for single_prompt_all_layers in batch_hidden_states_per_prompt:
            current_chunk_states.append(single_prompt_all_layers)

        # When a chunk is ready, put it on the queue for the save worker
        while len(current_chunk_states) >= chunk_size:
            states_to_save = current_chunk_states[:chunk_size]
            
            chunk_array = np.empty(len(states_to_save), dtype=object)
            for idx, state in enumerate(states_to_save):
                chunk_array[idx] = state

            chunk_file_path = f"{base_name}_part_{chunk_number}{ext}"
            
            # Put the data onto the queue for the saver thread to process
            save_queue.put((chunk_array, chunk_file_path, chunk_number))
            
            current_chunk_states = current_chunk_states[chunk_size:]
            chunk_number += 1
            
    # After the loop, queue any remaining hidden states
    if current_chunk_states:
        print("\nQueueing final remaining chunk...")
        final_chunk_array = np.empty(len(current_chunk_states), dtype=object)
        for idx, state in enumerate(current_chunk_states):
            final_chunk_array[idx] = state
        
        chunk_file_path = f"{base_name}_part_{chunk_number}{ext}"
        save_queue.put((final_chunk_array, chunk_file_path, chunk_number))

    # Signal the save worker to terminate by putting 'None' in the queue
    print("\nInference complete. Waiting for save worker to finish...")
    save_queue.put(None)
    
    # Wait for the saver thread to finish its job
    saver_thread.join()

    print(f"\nAll processing finished. {len(saved_chunk_paths)} files were created.")
    return saved_chunk_paths


def extract_dataset(dataset=0, output_dir = "output"):
    print("Extraction of the hidden states for jailbreak classification")
    

    model_name = MODEL_BASE
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    benign, jailbreak = []
    dataset = 0
    if dataset == 0:
        benign, jailbreak = get_dataset_malicious()
    elif dataset == 1:
        benign, jailbreak = get_dataset_jailbreak()
    else:
        benign,jailbreak = get_dataset_attack()

    torch.cuda.empty_cache()
    print("indices computed")
    model = AutoModel.from_pretrained(model_name).to(device)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    GPU_BATCH_SIZE = 4

    # IMPORTANT: Llama models often don't have a pad token, which is required for batching.
    # We can use the end-of-sentence token as the padding token.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("--- Model and tokenizer loaded successfully. ---",flush=True)    
    

    
    # Define the number of prompts you want to process from each category.


    PROMPT_BATCH_SIZE_PER_CATEGORY = 50
    os.makedirs(output_dir, exist_ok=True)
    
    datasets_to_process = {
        "benign": benign,
        "jailbreak" : jailbreak
    }
    
    
    for name, queries in datasets_to_process.items():
            print(f"\n--- Processing dataset: '{name}' ---", flush=True)
            
            # Calculate the number of batches needed for this category
            num_batches = (len(queries) + PROMPT_BATCH_SIZE_PER_CATEGORY - 1) // PROMPT_BATCH_SIZE_PER_CATEGORY
            
            for i in range(num_batches):

                

                # Define the start and end indices for the current batch of prompts
                start_index = i * PROMPT_BATCH_SIZE_PER_CATEGORY
                end_index = start_index + PROMPT_BATCH_SIZE_PER_CATEGORY
                prompt_batch = queries[start_index:end_index]
                
                print(f"Processing batch {i+1}/{num_batches} for '{name}' (prompts {start_index} to {end_index-1})")

                # Create a unique file path for this specific batch
                output_file_path = os.path.join(output_dir, f"{name}_batch_{i}.npy")
                         
            # Check if the output file for this batch already exists. If so, skip it.
                if os.path.exists(output_file_path):
                    print(f"Batch {i+1}/{num_batches} already exists. Skipping.")
                    continue
            
                # Process the batch of prompts and save the hidden states
                get_hidden_states_from_prompts_optimized(
                    model=model,
                    tokenizer=tokenizer,
                    prompts=prompt_batch,
                    output_file_path=output_file_path,
                    batch_size=GPU_BATCH_SIZE,
                )

    print("\n🎉 All tasks completed successfully!")

    return


if __name__ == "__main__":
    extract_dataset(dataset=1,output_dir="output_jailbreak")