
# Trait Simulation Tutorial


Authors: Sarah Ji, Janet Sinsheimer, Kenneth Lange

### Double check that you are using Julia version 1.0 or higher by checking the machine information


```julia
versioninfo()
```
# Add any missing packages needed for this tutorial:

You can also use the package manager to add the `TraitSimulation.jl` package by running the following link: 

```{julia}
pkg> add "https://github.com/sarah-ji/TraitSimulation.jl"
```


## Example of M variance components

Here for m = 10 random Variance Components, we generate m random covariance matrices, a random design matrix and p regression coefficients to illustrate the simulation of a d-dimensional response matrix for a sample of n = 1000 people.


```julia
using Random
Random.seed!(1234);
n = 1000   # no. observations
d = 2      # dimension of responses
m = 10      # no. variance components
p = 2      # no. covariates

# n-by-p design matrix
X = randn(n, p)

# p-by-d mean component regression coefficient
B = ones(p, d)  

# a tuple of m covariance matrices
# a tuple of m covariance matrices
V = ntuple(x -> zeros(n, n), m) 
for i = 1:m-1
  V_i = [j ? i ? i * (n - j + 1) : j * (n - i + 1) for i in 1:n, j in 1:n]
  copy!(V[i], V_i * V_i')
end
copy!(V[m], Diagonal(ones(n))) # last covarianec matrix is idendity

# a tuple of m d-by-d variance component parameters
? = ntuple(x -> zeros(d, d), m) 
for i in 1:m
  ?_i = [j ? i ? i * (d - j + 1) : j * (d - i + 1) for i in 1:d, j in 1:d]
  copy!(?[i], ?_i' * ?_i)
end

Random_VCM_Trait = DataFrame(VCM_simulation(X, B, V, ?), [:SimTrait1, :SimTrait2])
```

```julia
@benchmark VCM_simulation(X, B, V, ?)
```
    BenchmarkTools.Trial: 
      memory estimate:  152.81 MiB
      allocs estimate:  180
      --------------
      minimum time:     199.118 ms (51.36\% GC)
      median time:      204.352 ms (51.57\% GC)
      mean time:        205.967 ms (51.21\% GC)
      maximum time:     226.747 ms (48.48\% GC)
      --------------
      samples:          25
      evals/sample:     1

# Comparing with the MatrixNormal from the distributions package for a single variance component, we beat! 

```julia
@benchmark LMM_trait_simulation(X*B, VarianceComponent(?[1], V[1]))
```
    BenchmarkTools.Trial: 
      memory estimate:  15.31 MiB
      allocs estimate:  17
      --------------
      minimum time:     9.283 ms (0.00\% GC)
      median time:      12.040 ms (20.83\% GC)
      mean time:        11.598 ms (17.30\% GC)
      maximum time:     15.605 ms (24.83\% GC)
      --------------
      samples:          431
      evals/sample:     1

```julia
using Distributions
function MN_J(X, B, V, ?)
    return(rand(MatrixNormal(X*B, V, ?)))
end

@benchmark MN_J(X, B, V[1], ?[2])
```

    BenchmarkTools.Trial: 
      memory estimate:  15.35 MiB
      allocs estimate:  22
      --------------
      minimum time:     9.341 ms (0.00\% GC)
      median time:      12.143 ms (20.88\% GC)
      mean time:        11.733 ms (17.13\% GC)
      maximum time:     15.821 ms (23.28\% GC)
      --------------
      samples:          426
      evals/sample:     1



## Citations: 

[1] Lange K, Papp JC, Sinsheimer JS, Sripracha R, Zhou H, Sobel EM (2013) Mendel: The Swiss army knife of genetic analysis programs. Bioinformatics 29:1568-1570.`


[2] OPENMENDEL: a cooperative programming project for statistical genetics.
[Hum Genet. 2019 Mar 26. doi: 10.1007/s00439-019-02001-z](https://www.ncbi.nlm.nih.gov/pubmed/?term=OPENMENDEL).

