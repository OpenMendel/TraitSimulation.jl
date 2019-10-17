## Example 2: Linear Mixed Model

In this example we show how to generate data so that the related individuals have correlated trait values even after we account for the effect of a SNP, a combination of SNP's or other fixed effects.


We simulate two correlated Normal Traits controlling for family structure, location = $\mu$ and scale = $\mathbf\Sigma$. 
The corresponding bivariate variance covariance matrix as specified Mendel Option 28e, $\mathbf{\Sigma}$, is generated here.

## The Variance Covariance Matrix

Recall : $E(\mathbf{GRM}) = \Phi$

We use the [SnpArrays.jl](https://github.com/OpenMendel/SnpArrays.jl) package to find an estimate of the Kinship ($\Phi$), the Genetic Relationship Matrix (GRM). 

We will use the same values of $\textbf{GRM, V_a, and V_e}$ in the bivariate covariance matrix for both the mixed effect example and for the rare variant example.

Note that the residual covariance among two relatives is the additive genetic variance, $\textbf{V_a}$, times twice the kinship coefficient, $\Phi$. The kinship matrix is derived from the genetic relationship matrix $\textbf{GRM}$ across the common SNPs with minor allele frequency at least 0.05.

For convenience we use the common assumption that the residual covariance among two relatives can be captured by the additive genetic variance times twice the kinship coefficient. However, if you like you can specify your own variance components and their design matrices as long as they are positive semi definite using the `@vc` macro demonstrated in this example. We run this simulation 1000 times, and store the simulation results in a vector of DataFrames.


## Multiple Correlated Traits: (Mendel Example 28e Simulation)

$$Y \sim N(\mathbf{\mu},\Sigma  = V_{a} \otimes (2GRM) + V_{e} \otimes I_{n}) ,$$

$\text{vec}(Y) \sim \text{Normal}(X B, \Sigma_1 \otimes V_1 + \cdots + \Sigma_m \otimes V_m),$

where $\otimes$ is the [Kronecker product](https://en.wikipedia.org/wiki/Kronecker_product).

In this model, **data** is represented by  

* `Y`: `n x d` response matrix  
* `X`: `n x p` covariate matrix  
* `V=(V1,...,Vm)`: a tuple `m` `n x n` covariance matrices  

and **parameters** are  

* `B`: `p x d` mean parameter matrix  
* `Σ=(Σ1,...,Σm)`: a tuple of `m` `d x d` variance components  



```julia
Multiple_LMM_traits_model = LMMTrait(mean_formulas, X, variance_formula)
Simulated_LMM_Traits = simulate(Multiple_LMM_traits_model)
```
