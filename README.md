# The Geometry of Tokens in Internal Representations of Large Language Models
Source code for the paper: 'The Geometry of Tokens in Internal Representations of Large Language Models'

## Description

In this project, we analyze the geometry of tokens in the hidden layers of large language Models
using cosine similarity, intrinsic dimension (ID), and neighborhood overlap (NO). Given a prompt,
this is done in roughly two steps -

1. Extract the internal representations of the tokens -  We use the
[hidden states](https://huggingface.co/docs/transformers/v4.45.2/en/internal/generation_utils#generate-outputs)
variable from the [Transformers](https://huggingface.co/docs/transformers/index) library on Hugging Face.

2. Calculating the observables - For each hidden layer, we consider the point cloud formed
by the token representation at that layer. On this point cloud, we calculate
the cosine similarity, intrinsic dimension (ID), and neighborhood overlap (NO).
Specifically, we use the
[Generalized Ratio Intrinsic Dimension Estimator (GRIDE)](https://www.nature.com/articles/s41598-022-20991-1)
to estimate the intrinsic dimension implemented using the
[DADApy library](https://github.com/sissa-data-science/DADApy).

## Models
The list of models currently supported -
1. [Llama-3-8B](https://huggingface.co/meta-llama/Meta-Llama-3-8B)
2. [Mistral-7B-v0.1](https://huggingface.co/mistralai/Mistral-7B-v0.1)
3. [Pythia-6.9B](https://huggingface.co/EleutherAI/pythia-6.9b)


## Dataset
Currently we use the prompts from [Pile-10K](https://huggingface.co/datasets/NeelNanda/pile-10k).
We filter only prompts with atleast `1024` tokens according to the tokenization schemes
of all the above models. This results in `2244` prompts after filtering.
The indices of the filtered prompts is stored in `filtered_indices.npy`

### Shuffling experiment
For the shuffling experiment, we consider `50` random prompts from the filtered dataset,
i.e. we choose `50` prompts from the `2242` prompts. The indices of these prompts are stored in 
`subset_indices.npy`. 

## References

- [DADApy](https://github.com/sissa-data-science/DADApy)
- [GRIDE](https://www.nature.com/articles/s41598-022-20991-1)
- [Transformers Library](https://huggingface.co/docs/transformers/index)
