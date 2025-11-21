# Quantifying Semantic Disruption in Token Shuffling

## Experiment

To calibrate the semantic impact of our shuffling methodology, we computed BLEU and BERTScore metrics comparing shuffled prompts to their original versions. The analysis uses 50 prompts from Pile-10K with the Llama-3-8B tokenizer.

Our shuffling operates at 6 degrees (0-5) where block size = 1024 / (4^index):
- **Index 0**: No shuffling (block size = 1024 tokens)
- **Index 5**: Complete shuffling (block size = 1 token)

## Results

| Shuffle Index | BLEU [mean (std)] | BERTScore F1 [mean (std)] |
|----------------|-------------------|---------------------------|
| 0 (none)       | 1.00 (0.00)       | 1.00 (0.00)              |
| 1              | 0.99 (0.00)       | 0.90 (0.05)              |
| 2              | 0.97 (0.01)       | 0.88 (0.03)              |
| 3              | 0.86 (0.04)       | 0.86 (0.03)              |
| 4              | 0.42 (0.11)       | 0.81 (0.02)              |
| 5 (full)       | 0.03 (0.06)       | 0.76 (0.03)              |

The results show a consistent drop in both BLEU and BERTScore across shuffle degrees. BLEU drops from 1.00 to 0.03, while BERTScore decreases from 1.00 to 0.76, quantifying the semantic disruption caused by token shuffling at each degree.

## Internal Representations
