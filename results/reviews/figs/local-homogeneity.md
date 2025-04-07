# Checking for local homogenity

In our response to Reviewer 2UDe, we noted that estimators such as TwoNN and GRIDE rely on the assumption of local homogeneity, 
that is, nearby points are assumed to be uniformly sampled from d-dimensional balls.
In this note, we provide evidence that token representations meet a necessary condition for this assumption to hold in the 
context of the TwoNN estimator.

In [1], where the TwoNN estimator is introduced, the authors show that local homogeneity leads to a linear relationship between quantities
related to the log-ratios $\left(\mu = \frac{r_2}{r_1}\right)$ and the empirical cumulative distribution . In particular, the linear relationship is between
$\log(\mu)$ and $-\log(1 - F^{emp}(\mu_i))$. For a more elaborate explanation, please refer to the section "A Two Nearest Neighbors estimator for intrinsic dimension" in [1].
The intrinsic dimension is then estimated as the slope of this line, as illustrated in Figure 1 of [1]. 

We check if this distribution results in a straight line in our case by performing this check on prompt 3218 for layer 11 in 
[this figure](https://anonymous.4open.science/r/token_geometry-D346/results/reviews/figs/local_homogenity_layer_11.png) and 
for all layers in [this figure](https://anonymous.4open.science/r/token_geometry-D346/results/reviews/figs/local_homogeneity_all_layers.png).
It can be seen from the above figures that this results in a distribution implying that the token representations satisfy a necessary
condition of the local homogeneity hypothesis.

The above plot uses neighbors with knn = 1 and 2, as required by the TwoNN estimator. 
A natural next step is to evaluate the distribution using knn = 2 and 4, consistent with our range scaling factor of 4. 
However, this results in a distribution that is not linear, suggesting that a more sophisticated test is needed.

[1] Facco, Elena & d’Errico, Maria & Rodriguez, Alex & Laio, Alessandro. (2017). Estimating the intrinsic dimension of datasets by a minimal neighborhood information. Scientific Reports. 7. 10.1038/s41598-017-11873-y. 
