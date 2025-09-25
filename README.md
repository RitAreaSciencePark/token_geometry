# The Intrinsic Dimension of Prompts in Internal Representations of Large Language Models
Source code for the paper: 'The Intrinsic Dimension of Prompts in Internal Representations of Large Language Models'

## Description

In this project, we analyze the geometry of tokens in the hidden layers of large language models
using intrinsic dimension (ID). We study its relation to the surprisal of next token prediction 
and entropy of the latent predictions obtained from 
[TunedLens](https://huggingface.co/spaces/AlignmentResearch/tuned-lens/tree/main/lens). Given a prompt,
this is done by extracting the internal representations of the tokens -  We use the
[hidden states](https://huggingface.co/docs/transformers/v4.45.2/en/internal/generation_utils#generate-outputs)
variable from the [Transformers](https://huggingface.co/docs/transformers/index) library on Hugging Face. 
Make sure you are logged into HuggingFace Hub and have access to the models. We calculate the intrinsic dimension
for each hidden layer, we consider the point cloud formed by the token representation at that layer.
On this point cloud, we calculate the intrinsic dimension (ID).
Specifically, we use the
[Generalized Ratio Intrinsic Dimension Estimator (GRIDE)](https://www.nature.com/articles/s41598-022-20991-1)
to estimate the intrinsic dimension implemented using the
[DADApy library](https://github.com/sissa-data-science/DADApy). This is done by running the following example script
```
      python src/extract_id.py --input_dir results --model_name meta-llama/Meta-Llama-3-8B  --method structured
```
The above script generates the intrinsic dimension curve for `2244` (unshuffled) prompts for `Llama-3-8B` model. 
The ID curves are then stored in `results/Pile-Structured/meta-llama/Meta-Llama-3-8B/gride.npy`.
To run the code for the shuffled experiment, set `method` to `shuffled`. 

## Dataset
Currently we use the prompts from [Pile-10K](https://huggingface.co/datasets/NeelNanda/pile-10k).
We filter only prompts with atleast `1024` tokens according to the tokenization schemes
of all the above models. This results in `2244` prompts after filtering.
The indices of the filtered prompts is stored in `filtered_indices.npy`. We choose 50 random prompts
for the shuffle experiment in Figure 1 and their indices are stored in `subset_indices.npy`

## Reproducibility
The script to reproduce the plots in the paper is given in the [post_process](post_process) folder.
All experiments were run on an NVIDIA A100 GPU with 64 GB memory. 

## References

- [DADApy](https://github.com/sissa-data-science/DADApy)
- [GRIDE](https://www.nature.com/articles/s41598-022-20991-1)
- [Transformers Library](https://huggingface.co/docs/transformers/index)
