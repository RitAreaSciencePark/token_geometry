#!/usr/bin/env python3
"""
Compute semantic similarity scores (BLEU and BERTScore) for shuffled prompts.

This script quantifies the semantic disruption caused by token shuffling at different
degrees, providing quantitative measures to calibrate shuffle severity.

Usage:
    python reviews/compute_semantic_scores.py \
        --model_name meta-llama/Meta-Llama-3-8B \
        --output_dir reviews
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch
import numpy as np
import argparse
import json
from transformers import AutoTokenizer
from datasets import load_dataset
from evaluate import load
from tqdm import tqdm
from huggingface_hub import login

from src.extract_id import shuffle_tokens

os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.set_grad_enabled(False)
print("Disabled automatic differentiation")

# Login to HuggingFace (optional, may hit rate limits)
hf_token = os.environ.get('HF_TOKEN')
if hf_token:
    try:
        login(token=hf_token)
        print("Logged into HuggingFace")
    except Exception as e:
        print(f"Warning: HF login failed (rate limit or network issue): {e}")
        print("Continuing with cached models...")


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Compute BLEU and BERTScore for shuffled prompts"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="meta-llama/Meta-Llama-3-8B",
        help="Model name for tokenizer"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=1024,
        help="Maximum sequence length"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="reviews/results",
        help="Output directory for results"
    )
    parser.add_argument(
        "--num_shuffle_degrees",
        type=int,
        default=6,
        help="Number of shuffle degrees (0 to num_shuffle_degrees-1)"
    )

    args = parser.parse_args()
    print("Input arguments:\n", json.dumps(vars(args), indent=4, separators=(",", ":")))
    return args


def compute_scores_for_prompt(
    original_ids: torch.Tensor,
    tokenizer,
    bleu_metric,
    bertscore_metric,
    num_shuffle_degrees: int
):
    """
    Compute BLEU and BERTScore for a single prompt across all shuffle degrees.

    Parameters
    ----------
    original_ids : torch.Tensor
        Original token IDs for the prompt (shape: 1 x seq_len)
    tokenizer : AutoTokenizer
        Tokenizer for decoding
    bleu_metric : evaluate.Metric
        BLEU metric instance
    bertscore_metric : evaluate.Metric
        BERTScore metric instance
    num_shuffle_degrees : int
        Number of shuffle degrees to test (0=unshuffled, max=fully shuffled)

    Returns
    -------
    dict
        Dictionary with keys 'bleu', 'bertscore_precision', 'bertscore_recall', 'bertscore_f1'
        Each value is a list of length num_shuffle_degrees
    """
    original_text = tokenizer.decode(original_ids[0], skip_special_tokens=False)

    scores = {
        'bleu': [],
        'bertscore_precision': [],
        'bertscore_recall': [],
        'bertscore_f1': []
    }

    for shuffle_idx in range(num_shuffle_degrees):
        # Apply shuffling using existing function from extract_id.py
        # shuffle_idx 0: no shuffle (block_size = 1024)
        # shuffle_idx 5: fully shuffled (block_size = 1)
        shuffled_ids = shuffle_tokens(original_ids.clone(), shuffle_idx)
        shuffled_text = tokenizer.decode(shuffled_ids[0], skip_special_tokens=False)

        # Compute BLEU score
        bleu_result = bleu_metric.compute(
            predictions=[shuffled_text],
            references=[[original_text]]
        )
        scores['bleu'].append(bleu_result['bleu'])

        # Compute BERTScore
        bert_result = bertscore_metric.compute(
            predictions=[shuffled_text],
            references=[original_text],
            lang="en"
        )
        scores['bertscore_precision'].append(bert_result['precision'][0])
        scores['bertscore_recall'].append(bert_result['recall'][0])
        scores['bertscore_f1'].append(bert_result['f1'][0])

    return scores


def main():
    args = parse_arguments()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load tokenizer
    print(f"Loading tokenizer for {args.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # Load metrics
    print("Loading BLEU and BERTScore metrics...")
    bleu_metric = load("bleu")
    bertscore_metric = load("bertscore")

    # Load dataset
    print("Loading Pile-10K dataset...")
    ds = load_dataset("NeelNanda/pile-10k")['train']
    sequences = ds['text']

    # Load subset indices (50 prompts for shuffle experiment)
    subset_indices = np.load('subset_indices.npy')
    subset_sequences = [sequences[idx] for idx in subset_indices]
    num_prompts = len(subset_sequences)

    print(f"Processing {num_prompts} prompts from subset_indices.npy with {args.num_shuffle_degrees} shuffle degrees")

    # Initialize result storage
    results = {
        'bleu_scores': [],
        'bertscore_precision': [],
        'bertscore_recall': [],
        'bertscore_f1': [],
        'prompt_indices': subset_indices.tolist()
    }

    # Process each prompt
    for sequence in tqdm(subset_sequences, desc="Computing semantic scores"):
        # Tokenize the sequence
        inputs = tokenizer(
            sequence.strip(),
            add_special_tokens=False,
            return_tensors="pt",
            max_length=args.max_length,
            truncation=True
        )

        # Compute scores across all shuffle degrees
        prompt_scores = compute_scores_for_prompt(
            inputs['input_ids'],
            tokenizer,
            bleu_metric,
            bertscore_metric,
            args.num_shuffle_degrees
        )

        # Store results
        results['bleu_scores'].append(prompt_scores['bleu'])
        results['bertscore_precision'].append(prompt_scores['bertscore_precision'])
        results['bertscore_recall'].append(prompt_scores['bertscore_recall'])
        results['bertscore_f1'].append(prompt_scores['bertscore_f1'])

    # Convert to numpy arrays
    for key in ['bleu_scores', 'bertscore_precision', 'bertscore_recall', 'bertscore_f1']:
        results[key] = np.array(results[key])

    # Save results
    model_safe_name = args.model_name.replace('/', '_')
    output_file = os.path.join(
        args.output_dir,
        f"semantic_scores_{model_safe_name}_{num_prompts}prompts.npz"
    )
    np.savez_compressed(output_file, **results)

    print(f"\n✅ Results saved to {output_file}")
    print(f"BLEU scores shape: {results['bleu_scores'].shape}")
    print(f"BERTScore F1 shape: {results['bertscore_f1'].shape}")
    print(f"\n{'='*70}")
    print(f"SEMANTIC SCORE ANALYSIS ({num_prompts} prompts)")
    print(f"{'='*70}")
    print(f"{'Shuffle Degree':<20} {'BLEU':<25} {'BERTScore F1':<25}")
    print(f"{'-'*70}")
    for i in range(args.num_shuffle_degrees):
        bleu_mean = results['bleu_scores'][:, i].mean()
        bleu_std = results['bleu_scores'][:, i].std()
        bert_mean = results['bertscore_f1'][:, i].mean()
        bert_std = results['bertscore_f1'][:, i].std()
        print(f"{i:<20} {bleu_mean:.3f} ± {bleu_std:.3f}         {bert_mean:.3f} ± {bert_std:.3f}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
