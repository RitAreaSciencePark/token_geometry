# Rebuttal Plots

## Shuffling consistency check
Here, we perform two checks
1. Is the higher shuffled ID an artifact of noise? We perform the scale analysis for the shuffled prompts and show that they plateau around scaling = 4
2. When would a shuffled prompt be expected to show a lower ID than the normal sentence? Check for samples where this happens
### Scale analysis for shuffled prompts
![shuffled_scale_analysis](figs/shuffled_scale_analysis.png)

### Example of shuffled ID < unshuffled ID
![shuffled_less_than_unshuffled](figs/shuffled_less_than_unshuffled.png)

#### TAKEAWAY: At layer 15, prompt 3500 has a high normal ID and prompt 7505 has a lower shuffled ID. Document 3500 is a list of names and Document 7505 carries more context, thereby providing an example where Unshuffled ID (3500) > Shuffled ID (7505)

## ID estimation on random vocabulary tokens
Here we compare the ID of the prompt 3218, its shuffled version and 1024 tokens randomly sampled from the vocabulary. 
![vocab_comparison](figs/vocab_comparison.png)
#### Intrinsic dimension of the vocabulary > shuffled prompt > unshuffled prompt
The experiments here and the previous section above suggest that intrinsic dimension can be used as a measure of context in a prompt.
High intrinsic dimension suggests a lower level of context as seen in the vocabulary tokens, shuffled prompts, and PROMPT 3500 (which contained a list of names)

## Comparison with other estimators for PROMPT 3218
![id_estimators_comparison](figs/id_estimators_comparison.png)
We notice that TLE, ESS and GRIDE estimators have a similar ID profile across layers.
Another important factor is the computational advantage of GRIDE since the full experiment involves running across a large number (2244) of prompts.
Here is a table with the summary of number of point clouds processed per second.
| Estimator | Samples/sec (avg) |
|-----------|-------------------|
| **GRIDE** | 30.82             |
| **MOM**   | 14.06             |
| **MLE**   | 13.53             |
| **CorrInt** | 6.16           |
| **ESS**   | 2.97              |
| **PCA**   | 1.24              |
| **FisherS** | 1.21            |
| **TLE**   | 1.16              |
| **MADA**  | 0.92              |
