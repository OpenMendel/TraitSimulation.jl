
# TraitSimulation explores GPU computing

Author: Sarah Ji

In this notebook, we will explore some GPU capabilities of the Julia language in the CuArrays package, and present the potential for users to extend the functions in TraitSimulation.jl to accomodate advances in GPU computing. Although GPU computing is still under active development, it is fruitful for researchers to be aware of the most modern computing technologies available to users through open source options. 

We will first present the major bottle necks of the simulation algorithm and benchmark the optimized code (using BLAS) on CPU vs. on GPU. Then we will simulate the symmetric Bivariate trait using TraitSimulation on CPU first, then GPU for comparison. 

1. First the simulation algorithm first performs the Cholesky factors of the variance/covariance matrices and stores them using the @vc macro.
2. Then we transform the standard normal matrix using the Cholesky factors above

Note useres with many variance components or extremely large number of subjects may benefit most from exploring GPU. 
Additionally, since the default setting in the notebook is 1 thread, so I will only compare the unthreaded CPU code vs. the GPU code. 


```julia
versioninfo()
```

    Julia Version 1.4.0
    Commit b8e9a9ecc6 (2020-03-21 16:36 UTC)
    Platform Info:
      OS: Linux (x86_64-pc-linux-gnu)
      CPU: Intel(R) Core(TM) i9-9920X CPU @ 3.50GHz
      WORD_SIZE: 64
      LIBM: libopenlibm
      LLVM: libLLVM-8.0.1 (ORCJIT, skylake)


# Check available devices on this machine and show their capability


```julia
using CuArrays, CUDAdrv
using CuArrays.CURAND

for device in CuArrays.devices()
    @show capability(device)
end
```

    capability(device) = v"7.5.0"


We use simulated data and include the exploratory TraitSimulation on GPU code, in the file `exploring_gpu_simulation.jl`. We encourage users with the right NVIDIA GPU machine to explore these options for themselves


```julia
include("/home/sarahji/TraitSimulation.jl/src/exploring_gpu_simulation.jl");
```


```julia
using GLM, LinearAlgebra, Random, BenchmarkTools
using SnpArrays, Statistics

Random.seed!(1234)

function generateSPDmatrix(n)
    A = rand(n)
    m = 0.5 * (A * A')
    PDmat = m + (n * Diagonal(ones(n)))
end


function generateRandomVCM(n::Int64, p::Int64, d::Int64, m::Int64)
    # n-by-p design matrix
    X = randn(n, p)

    # p-by-d mean component regression coefficient for each trait
    B = rand(p, d)

    V = ntuple(x -> zeros(n, n), m)
    for i = 1:m-1
      copy!(V[i], generateSPDmatrix(n))
    end
    copy!(V[end], Diagonal(ones(n))) # last covarianec matrix is identity

    # a tuple of m d-by-d variance component parameters
    Σ = ntuple(x -> zeros(d, d), m)
    for i in 1:m
      copy!(Σ[i], generateSPDmatrix(d))
    end
    return(X, B, Σ, V)
end


n = 5000 # number of people
p = 3   # number of fixed effects
d = 10   # number of traits
m = 10   # number of variance components

X, β, Σ, V = generateRandomVCM(n, p, d, m);
```

# 1. Cholesky Decomposition of Variance Components

The major bottleneck in the simulation process is the cholesky decomposition of the variance covariance matrices (variance components). We note that if users have many variance components will benefit from having the GPU option available to their software. 

First we compute the Cholesky Decomposition of a single nxn matrix, V[1], then we use the @vc macro and make a GPU alternative to compare.

We see that for a 5000 by 5000 matrix, the GPU code is more than 30x faster than the cholesky decomposition computed on CPU.

### Cholesky on CPU for a single variance component (n x n)


```julia
# Cholesky on CPU for a single trait
@show size(V[1])
@benchmark cholesky((Symmetric($V[1])))
```
    size(V[1]) = (5000, 5000)

    BenchmarkTools.Trial: 
      memory estimate:  190.74 MiB
      allocs estimate:  6
      --------------
      minimum time:     277.191 ms (0.00% GC)
      median time:      292.412 ms (0.00% GC)
      mean time:        326.911 ms (1.62% GC)
      maximum time:     769.881 ms (4.69% GC)
      --------------
      samples:          16
      evals/sample:     1



### Cholesky on GPU for a single variance component (n x n)


```julia
@time V1_gpu = CuArray{Float32}(V[1]) # Move the Variance Covariance Matrix V from CPU to GPU environment
@benchmark cholesky((Symmetric($V1_gpu)))
```

      0.959381 seconds (1.94 M allocations: 191.373 MiB)

    BenchmarkTools.Trial: 
      memory estimate:  4.00 KiB
      allocs estimate:  118
      --------------
      minimum time:     10.111 ms (0.00% GC)
      median time:      10.289 ms (0.00% GC)
      mean time:        10.339 ms (0.34% GC)
      maximum time:     14.302 ms (19.20% GC)
      --------------
      samples:          484
      evals/sample:     1



We made this @vc macro to provide a more flexible input option for users, but we see that there is a cost for comfort!

The single cholesky is about 30 fold speed up but using the macro its only about 6x 
So we note that for maximum efficiency, when users have large datasets, they can bypass this and just store the cholesky factors necessary to perform the transformation.


```julia
vc_cpu = @vc Σ[1] ⊗ V[1] + Σ[2] ⊗ V[2] # make the vc object with just 2 variance components
trait = VCMTrait(X, β, [Σ...], [V...]) # make the trait object with all m = 10 variance components.

@benchmark vc_cpu = @vc $Σ[1] ⊗ $V[1] + $Σ[2] ⊗ $V[2]
```
    BenchmarkTools.Trial: 
      memory estimate:  381.47 MiB
      allocs estimate:  25
      --------------
      minimum time:     1.609 s (0.31% GC)
      median time:      1.801 s (3.48% GC)
      mean time:        2.100 s (3.78% GC)
      maximum time:     2.890 s (5.90% GC)
      --------------
      samples:          3
      evals/sample:     1




```julia
vc_gpu = @vc_gpu Σ[1] ⊗ V[1] + Σ[2] ⊗ V[2]
@benchmark vc_gpu = @vc_gpu $Σ[1] ⊗ $V[1] + $Σ[2] ⊗ $V[2]
```
    BenchmarkTools.Trial: 
      memory estimate:  16.78 KiB
      allocs estimate:  497
      --------------
      minimum time:     274.375 ms (0.00% GC)
      median time:      277.834 ms (0.00% GC)
      mean time:        292.647 ms (0.22% GC)
      maximum time:     350.880 ms (0.83% GC)
      --------------
      samples:          18
      evals/sample:     1



# 2. Matrix Normal Transformation by Cholesky Factors 

Each variance component will require it's own simulation from the standard normal distribution and the appropriate transformations by the cholesky factors of the row and column variances. 

We saw separately that the cholesky decomposition is sped up by an order of magnitude on GPU vs. on CPU in step 1.
Now, to complete the simulation we will perform step 2 to multiply on the left and right to transform the simulated trait to have the desired covariance.

There is a 30x fold speed up in JUST transforming the matrix normal using GPU.


```julia
function cpu_single_vc(trait, CholΣ, CholV)
    randn!(trait.Z)
    BLAS.trmm!('L', 'U', 'T', 'N', 1.0, CholV, trait.Z)
    BLAS.trmm!('R', 'U', 'N', 'N', 1.0, CholΣ, trait.Z)
    trait.Z
end

CholΣ = trait.vc[1].CholΣ # grab the cholesky factors of Σ (dxd) from step 1
CholV = trait.vc[1].CholV # grab the cholesky factor of V (nxn) from step 1

@benchmark cpu_single_vc($trait, $CholΣ, $CholV)
```
    BenchmarkTools.Trial: 
      memory estimate:  0 bytes
      allocs estimate:  0
      --------------
      minimum time:     19.655 ms (0.00% GC)
      median time:      20.967 ms (0.00% GC)
      mean time:        21.233 ms (0.00% GC)
      maximum time:     23.723 ms (0.00% GC)
      --------------
      samples:          236
      evals/sample:     1




```julia
function gpu_single_vc(Z_d, CholΣ, CholV)
    randn!(Z_d)
    CuArrays.CUBLAS.trmm!('L', 'U', 'T', 'N', 1.0, CholV, Z_d, similar(Z_d))
    CuArrays.CUBLAS.trmm!('R', 'U', 'N', 'N', 1.0, CholΣ, Z_d, similar(Z_d))
    Z_d
end

Z_d = CuArray{Float64}(trait.Z)
CholΣ_gpu = vc_gpu[1].CholΣ # grab the cholesky factors of Σ (dxd) from step 1
CholV_gpu = vc_gpu[1].CholV # grab the cholesky factor of V (nxn) from step 1

@benchmark gpu_single_vc($Z_d, $CholΣ_gpu, $CholV_gpu)
```

    BenchmarkTools.Trial: 
      memory estimate:  784 bytes
      allocs estimate:  29
      --------------
      minimum time:     16.213 μs (0.00% GC)
      median time:      1.061 ms (0.00% GC)
      mean time:        1.010 ms (0.08% GC)
      maximum time:     10.271 ms (38.50% GC)
      --------------
      samples:          4946
      evals/sample:     1



# Perform TraitSimulation

Here we just compare a single simulation on CPU vs. on GPU. 
This is the function that is continuously called in the power calculation. It writes over the field Y, the simulated results after aggregating the simulation results of the m variance components. (random effects) 

1. First we see the simulation for a single variance component
2. Then we see the simulation for the entire variance component set in trait.vc

### TraitSimulation for a Single Variance Component

We see this is roughly 20 times faster on GPU than on CPU for a single $V_{n \times n}, \Sigma_{d \times d}$


```julia
@benchmark TraitSimulation.simulate_matrix_normal!($trait.Z, $vc_cpu[1])
```
    BenchmarkTools.Trial: 
      memory estimate:  0 bytes
      allocs estimate:  0
      --------------
      minimum time:     27.357 ms (0.00% GC)
      median time:      32.112 ms (0.00% GC)
      mean time:        40.311 ms (0.00% GC)
      maximum time:     83.568 ms (0.00% GC)
      --------------
      samples:          124
      evals/sample:     1




```julia
Z_d = CuArray{Float64}(trait.Z)
@benchmark simulate_matrix_normal_gpu!($Z_d, $vc_gpu[1])
```
    BenchmarkTools.Trial: 
      memory estimate:  784 bytes
      allocs estimate:  29
      --------------
      minimum time:     17.176 μs (0.00% GC)
      median time:      1.060 ms (0.00% GC)
      mean time:        1.009 ms (0.00% GC)
      maximum time:     1.468 ms (0.00% GC)
      --------------
      samples:          4950
      evals/sample:     1



### TraitSimulation for the set of m Variance Components

We see above that the simulation for just a single variance component is roughly 30 times faster. 

Since we have m = 10 variance components in our model, we see that in this case the simulation on GPU is 100x faster.


```julia
Y = zero(trait.μ)
@benchmark TraitSimulation.VCM_trait_simulation($Y, $trait.Z, $trait.μ, $trait.vc)
```

    BenchmarkTools.Trial: 
      memory estimate:  0 bytes
      allocs estimate:  0
      --------------
      minimum time:     200.812 ms (0.00% GC)
      median time:      257.338 ms (0.00% GC)
      mean time:        266.100 ms (0.00% GC)
      maximum time:     626.675 ms (0.00% GC)
      --------------
      samples:          19
      evals/sample:     1




```julia
Y_d = zeros(CuArray{Float32}, size(trait.Z))
μ_d = zeros(CuArray{Float32}, size(trait.μ))
@benchmark VCM_trait_simulation_gpu($Y_d, $μ_d, $Z_d, $vc_gpu)
```

    BenchmarkTools.Trial: 
      memory estimate:  10.67 KiB
      allocs estimate:  283
      --------------
      minimum time:     67.968 μs (0.00% GC)
      median time:      2.134 ms (0.00% GC)
      mean time:        2.056 ms (0.00% GC)
      maximum time:     2.361 ms (0.00% GC)
      --------------
      samples:          2430
      evals/sample:     1


### Threading:

For users who wish to do multiple simulation runs simultaneously, we recommend to set the machinery to use threading. Users can check using the command: `Threads.nthreads()` to ensure multi-threading is on. [TraitSimulation](https://github.com/OpenMendel/TraitSimulation.jl/blob/master/src/TraitSimulation.jl#L188) will automatically use the Threading option for multiple TraitSimulation. To set the number of threads, users should follow the documentation on [Threads.jl](https://docs.julialang.org/en/v1/base/multi-threading/) and ensure before starting Julia to specify the desired number of threads using ` export JULIA_NUM_THREADS=4`.

For users who are using the Threading option and are seeing some variation in the benchmarking results, make sure that the julia number of threads and the BLAS number of threads are not confusing one another and specify the command: `LinearAlgebra.BLAS.set_num_threads(1)` to make simulations more consistent.


