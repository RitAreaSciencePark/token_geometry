import numpy as np

from transformers import AutoModel, AutoTokenizer,AutoModelForCausalLM
import torch
import os
import json
from tqdm import tqdm
import queue
import threading
from datasets import load_dataset, Dataset
from tuned_lens.nn.lenses import TunedLens
from compute_guard_models import  get_dataset_malicious,get_dataset_jailbreak,get_dataset_attack

MODEL_BASE = "meta-llama/Meta-Llama-3-8B"  #


def extract_entropies_from_input_ids(input_ids, model, lens, tokenizer):
    with torch.no_grad():
        input_ids_th = input_ids.clone().detach().to(model.device)
        outputs = model(input_ids_th, output_hidden_states=True)
        stream = list(outputs.hidden_states)
        entropy, hfe, expected_energy = [], [], []
      
        for i, h in enumerate(stream[:-1]):
            logits = lens.forward(h, i).squeeze()
            probs = torch.nn.functional.softmax(logits, dim=-1)
            energies = logits - torch.max(logits, axis=-1, keepdims=True).values
            hfe.append(-torch.log(torch.exp(energies).sum(axis=-1)).mean().cpu().numpy())
            entropy.append(-(torch.sum(probs * torch.log(probs + 1e-12), axis=-1)).mean().cpu().numpy())
            expected_energy.append(torch.sum(probs * (-energies), axis=-1).mean().cpu().numpy())
        
        logits = outputs.logits.squeeze()
        probs = torch.nn.functional.softmax(logits, dim=-1)
        energies = logits - torch.max(logits, axis=-1, keepdims=True).values
        hfe.append(-torch.log(torch.exp(energies).sum(axis=-1)).mean().cpu().numpy())
        entropy.append(-(torch.sum(probs * torch.log(probs), axis=-1)).mean().cpu().numpy())
        expected_energy.append(torch.sum(probs * (-energies), axis=-1).mean().cpu().numpy())
        
        return {"entropy": np.array(entropy), "free_energy": np.array(hfe), "energy": np.array(expected_energy)}

def extract_entropies(sequence, model, lens, tokenizer):
    input_ids = tokenizer.encode(sequence.strip(), add_special_tokens=False, return_tensors="pt", truncation=True)
    return extract_entropies_from_input_ids(input_ids, model, lens, tokenizer)


if __name__ == "__main__":

    print("Generating entropies for malicious prompts in range 500-1000")


    model_name = MODEL_BASE
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    benign, jailbreak = []
    dataset = 0
    if dataset == 0:
        benign, jailbreak = get_dataset_malicious()
    elif dataset == 1:
        benign, jailbreak = get_dataset_jailbreak()
    else:
        benign,jailbreak = get_dataset_attack()


    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

    lens = TunedLens.from_model_and_pretrained(model,"meta-llama/Meta-Llama-3-8B").to(device)
    print(f"Found {len(benign)} benign prompts and {len(jailbreak)} jailbreak prompts.")

    # --- Processing Benign Prompts ---
    print("\nProcessing benign prompts...")
    benign_entropies = []
    for prompt in tqdm(benign, desc="Benign Prompts"):
        results = extract_entropies(prompt, model, lens, tokenizer)
        benign_entropies.append(results["entropy"])
    # Convert to NumPy array and transpose to get shape [#layers, #prompts]
    if benign_entropies:
        benign_entropies_np = np.array(benign_entropies).T
        print(f"Shape of benign entropies array: {benign_entropies_np.shape}")
        np.save("path_benign.npy", benign_entropies_np)
        print("Saved benign entropies to path_benign.npy")
    else:
        print("No benign prompts were processed.")
    # --- Processing Jailbreak Prompts ---
    print("\nProcessing jailbreak prompts...")
    jailbreak_entropies = []
    for prompt in tqdm(jailbreak, desc="Jailbreak Prompts"):
        results = extract_entropies(prompt, model, lens, tokenizer)
        jailbreak_entropies.append(results["entropy"])
    # Convert to NumPy array and transpose to get shape [#layers, #prompts]
    if jailbreak_entropies:
        jailbreak_entropies_np = np.array(jailbreak_entropies).T
        print(f"Shape of jailbreak entropies array: {jailbreak_entropies_np.shape}")
        np.save("path_jailbreak.npy", jailbreak_entropies_np)
        print("Saved jailbreak entropies to path_benign.npy")
    else:
        print("No jailbreak prompts were processed.")
    print("\nScript finished.")
