
# Trait Simulation: Prototyping Analysis Methods (MendelIHT)

Authors: Sarah Ji, Benjamin Chu, Janet Sinsheimer, Kenneth Lange, Hua Zhou

This software package, TraitSimuliation.jl addresses the need for simulated trait data in genetic analyses.  This package generates data sets that will allow researchers to accurately check the validity of programs and to calculate power for their proposed studies. 

In this notebook we show how to the `TraitSimulation.jl` package can be used to validate the results of the newest OpenMendel analysis package, `MendelIHT.jl`.

First we will generate the desired non-genetic covariates for our examples, sex and age from the `GLM.jl` package. Then we will make the appropriate calls to OpenMendel packages SnpArrays.jl, MendelIHT.jl. and TraitSimulation.jl to construct the desired genetic model, simulate from it and test the iterative hard thresholding model.

For the following three distributions, we simulate traits and validate the results using MendelIHT.


1. Normal Trait
2. Poisson Trait
3. Negative Binomial Trait


Double check that you are using Julia version 1.0 or higher by checking the machine information


```julia
versioninfo()
```

    Julia Version 1.2.0
    Commit c6da87ff4b (2019-08-20 00:03 UTC)
    Platform Info:
      OS: macOS (x86_64-apple-darwin18.6.0)
      CPU: Intel(R) Core(TM) i7-7700HQ CPU @ 2.80GHz
      WORD_SIZE: 64
      LIBM: libopenlibm
      LLVM: libLLVM-6.0.1 (ORCJIT, skylake)



```julia
using Random, DataFrames, LinearAlgebra
using SnpArrays, TraitSimulation, GLM, StatsBase, MendelIHT
using Revise
revise()

Random.seed!(123);
```

Simulate Non-Genetic Covariates


```julia
n = 5000
p = 10000
s = 5

# simulate non genetic covariates
intercept = ones(n, 1) 
sex = rand(Bernoulli(0.51), n)
age = zscore(rand(Normal(45, 8), n))
non_genetic_cov =  [sex age]
β_non_gen = rand(size(non_genetic_cov, 2))
```




    2-element Array{Float64,1}:
     0.7373240898122129
     0.4314238505987109



Simulate Genetic Covariates 

First we will generate the entire snparray of p = 10,000 snps for n = 1000 people. Then for s = 10 causal snps, we will convert them to Float64 integers for simulation.


```julia
# generate entire snparray
x = simulate_random_snparray(n, p, "tmp.bed")
xbm = SnpBitMatrix{Float64}(x, model=ADDITIVE_MODEL, center=true, scale=true); 
k = 5
# generate snparray with only causal snps
causal_idx = rand(1:10000, k) # these are the true indices
causal_effect_sizes = rand(k) # these are the true effect sizes

x_causal = convert(Matrix{Float64}, @view(x[:, causal_idx]), center = true, scale = true);
```

## Simulate the GLM Trait

Use TraitSimulation.jl to construct the desired model and simulate from it. For the following three distributions, we simulate traits and validate the results using MendelIHT.


1. Normal Trait
2. Poisson Trait
3. Negative Binomial Trait

### Testing Normal Trait


```julia
distribution = Normal()
link = IdentityLink()
# revise()
GLM_model = GLMTrait(non_genetic_cov, β_non_gen, x_causal, causal_effect_sizes, distribution, link)
y = simulate(GLM_model)
```




    5000-element Array{Float64,1}:
      0.9185868876881607    
     -0.9518262679882143    
     -0.33841827510588013   
      2.049817083313287     
      1.955913151958258     
      0.00024397836600864933
     -0.6593742370456934    
     -0.28351854873874294   
      0.3185448435991667    
      0.30756420729346534   
     -0.4628928677570747    
      1.292210040257612     
      0.06891346416538825   
      ⋮                     
     -1.0982567760970272    
     -0.016491104287002845  
      0.9345002196941151    
      0.14027327315163013   
     -2.20745242227931      
      1.8222300720529194    
      2.9752072523810957    
      0.4504020749944919    
     -1.7701081883560879    
     -0.3661636604161142    
      2.5102925366869027    
      0.9083339384177506    


# Run Mendel IHT

```julia
s_guess = 7
@show β_non_gen
@show truth = causal_idx 
@show causal_effect_sizes
result = L0_reg(x, xbm, non_genetic_cov, y, 50, s_guess, Normal(), IdentityLink(), debias = false, max_iter = 500, verbose= false)
```

    β_non_gen = [0.7373240898122129, 0.4314238505987109]
    truth = causal_idx = [184, 4492, 5382, 161, 2773]
    causal_effect_sizes = [0.6877212499518375, 0.7610202033321085, 0.6111539989474444, 0.7062198720454749, 0.06323470301651035]





    
    IHT estimated 5 nonzero SNP predictors and 2 non-genetic predictors.
    
    Compute time (sec):     2.9930989742279053
    Final loglikelihood:    -7115.458807213961
    Iterations:             10
    
    Selected genetic predictors:
    5×2 DataFrame
    │ Row │ Position │ Estimated_β │
    │     │ [90mInt64[39m    │ [90mFloat64[39m     │
    ├─────┼──────────┼─────────────┤
    │ 1   │ 161      │ 0.702183    │
    │ 2   │ 184      │ 0.68639     │
    │ 3   │ 4492     │ 0.75284     │
    │ 4   │ 5382     │ 0.595549    │
    │ 5   │ 5867     │ 0.0581857   │
    
    Selected nongenetic predictors:
    2×2 DataFrame
    │ Row │ Position │ Estimated_β │
    │     │ [90mInt64[39m    │ [90mFloat64[39m     │
    ├─────┼──────────┼─────────────┤
    │ 1   │ 1        │ 0.715045    │
    │ 2   │ 2        │ 0.428766    │



### Testing Poisson Trait



```julia
distribution2 = Poisson()
link2 = LogLink()

non_genetic_cov =  [sex age]
β_non_gen = [0.1, 0.002]
```




    2-element Array{Float64,1}:
     0.1  
     0.002




```julia
GLM_model2 = GLMTrait(non_genetic_cov, β_non_gen, x_causal, causal_effect_sizes, Poisson, link2)
```




    Generalized Linear Model
      * response distribution: UnionAll
      * link function: LogLink
      * sample size: 5000




```julia
x = simulate_random_snparray(n, p, "tmp.bed")
xbm = SnpBitMatrix{Float64}(x, model=ADDITIVE_MODEL, center=true, scale=true);
k = 5
causal_idx = rand(1:10000, k) # these are the true indices
causal_effect_sizes = rand(k)
x_causal = convert(Matrix{Float64}, @view(x[:, causal_idx]), center = true, scale = true);
```


```julia
# construct the model 
GLM_model_p2 = GLMTrait(non_genetic_cov, β_non_gen, x_causal, causal_effect_sizes, Poisson(), link2)
```




    Generalized Linear Model
      * response distribution: Poisson
      * link function: LogLink
      * sample size: 5000




```julia
# simulate the trait
y_poisson2 = simulate(GLM_model_p2)
```




    5000-element Array{Int64,1}:
     0
     3
     0
     0
     2
     0
     0
     0
     0
     7
     1
     0
     2
     ⋮
     6
     1
     2
     4
     4
     2
     0
     0
     1
     3
     1
     0




```julia
s_guess = 7
result2 = L0_reg(x, xbm, non_genetic_cov, Float64.(y_poisson2), 50, s_guess, Poisson(), LogLink(), debias = false, max_iter = 500, verbose= false)
```




    
    IHT estimated 6 nonzero SNP predictors and 1 non-genetic predictors.
    
    Compute time (sec):     8.748939990997314
    Final loglikelihood:    -6742.100029722215
    Iterations:             32
    
    Selected genetic predictors:
    6×2 DataFrame
    │ Row │ Position │ Estimated_β │
    │     │ [90mInt64[39m    │ [90mFloat64[39m     │
    ├─────┼──────────┼─────────────┤
    │ 1   │ 400      │ 0.405941    │
    │ 2   │ 2943     │ 0.0423735   │
    │ 3   │ 3749     │ 0.036553    │
    │ 4   │ 4656     │ 0.53714     │
    │ 5   │ 7563     │ 0.722796    │
    │ 6   │ 8016     │ 0.239444    │
    
    Selected nongenetic predictors:
    1×2 DataFrame
    │ Row │ Position │ Estimated_β │
    │     │ [90mInt64[39m    │ [90mFloat64[39m     │
    ├─────┼──────────┼─────────────┤
    │ 1   │ 1        │ 0.0952795   │



### Testing Negative Binomial Trait


```julia
# simulate the trait
distribution = NegativeBinomial()
link = LogLink()

GLM_model = GLMTrait(non_genetic_cov, β_non_gen, x_causal, causal_effect_sizes, distribution, link)
y_negbinomial = simulate(GLM_model)
```




    5000-element Array{Int64,1}:
      1
      1
      0
      0
      2
      1
      1
      1
      0
      1
      0
      1
      0
      �
     15
      2
      3
      2
     13
      1
      0
      0
      1
      1
      0
      0




```julia
# run IHT
s_guess = 7
@show β_non_gen
@show truth = causal_idx 
@show causal_effect_sizes
result = L0_reg(x, xbm, non_genetic_cov, Float64.(y_negbinomial), 50, s_guess, NegativeBinomial(), link, debias = false, use_maf = false, max_iter = 500, verbose= false)
```

    β_non_gen = [0.1, 0.002]
    truth = causal_idx = [4656, 9394, 400, 8016, 7563]
    causal_effect_sizes = [0.5264504200595785, 0.008085205500707904, 0.4110283000164554, 0.22887974718537518, 0.7299946760225595]





    
    IHT estimated 6 nonzero SNP predictors and 1 non-genetic predictors.
    
    Compute time (sec):     4.818864107131958
    Final loglikelihood:    -7579.647514751706
    Iterations:             17
    
    Selected genetic predictors:
    6×2 DataFrame
    │ Row │ Position │ Estimated_β │
    │     │ [90mInt64[39m    │ [90mFloat64[39m     │
    ├─────┼──────────┼─────────────┤
    │ 1   │ 400      │ 0.418851    │
    │ 2   │ 1118     │ 0.0727529   │
    │ 3   │ 3367     │ -0.0822334  │
    │ 4   │ 4656     │ 0.50584     │
    │ 5   │ 7563     │ 0.737677    │
    │ 6   │ 8016     │ 0.193797    │
    
    Selected nongenetic predictors:
    1×2 DataFrame
    │ Row │ Position │ Estimated_β │
    │     │ [90mInt64[39m    │ [90mFloat64[39m     │
    ├─────┼──────────┼─────────────┤
    │ 1   │ 1        │ 0.140459    │



## Citations: 

[1] Lange K, Papp JC, Sinsheimer JS, Sripracha R, Zhou H, Sobel EM (2013) Mendel: The Swiss army knife of genetic analysis programs. Bioinformatics 29:1568-1570.`


[2] OPENMENDEL: a cooperative programming project for statistical genetics.
[Hum Genet. 2019 Mar 26. doi: 10.1007/s00439-019-02001-z](https://www.ncbi.nlm.nih.gov/pubmed/?term=OPENMENDEL).

[3] Benjamin B. Chu, Kevin L. Keys, Christopher A. German, Hua Zhou, Jin J. Zhou, Janet S. Sinsheimer, Kenneth Lange. Iterative Hard Thresholding in GWAS: Generalized Linear Models, Prior Weights, and Double Sparsity. bioRxiv doi:10.1101/697755

