## Example 1: Generalized Linear Model
In this example we show how to generate multiple traits from GLM's with a genetic variant in the fixed effects, but no residual familial correlation.


In example (1a) we simulate a single independent Normal trait with simulation parameters: $\mu = 20 + 3*sex - 1.5*locus$, $\sigma^{2} = 2$. By default, without specifying a number of replicates for the user specified GLM (like this example), the `simulate` function returns a single simulated trait.


This example simulates a case where three snps have fixed effects on the trait. Any apparent genetic correlation between relatives for the trait is due to the effect of these snps, so once these effects of these snps are modelled there should be no residual correlation among relatives. Note that by default, individuals with missing genotype values will have missing phenotype values, unless the user specifies the argument `impute = true` in the convert function above.


### Single Trait
$$Y \sim N(\mu, \sigma^{2})$$

In example (1a) we simulate a $\textbf{SINGLE INDEPENDENT NORMAL TRAIT}$, with simulation parameters: $\mu = 20 + 3*sex - 1.5*locus$, $\sigma^{2} = 2$


```julia
mean_formula = "20 + 3(sex) - 1.5(locus)"
GLM_trait_model = GLMTrait(mean_formula, X, NormalResponse(2), IdentityLink())
Simulated_GLM_trait = simulate(GLM_trait_model)
```

```@autodocs
Modules = [TraitSimulation]
```
