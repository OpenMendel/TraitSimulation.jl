
# Trait Simulation Tutorial

Authors: Sarah Ji, Janet Sinsheimer, Kenneth Lange

```@meta
CurrentModule = TraitSimulation
```

```@autodocs
Modules = [TraitSimulation]
```

In this notebook we show how to use the `TraitSimulation.jl` package to simulate traits from genotype data from unrelateds or families on user-specified Generalized Linear Models (GLMs) or Linear Mixed Models (LMMs), respectively.

The data we will be using is from the Mendel version 16[1] sample files. The data is described in examples under Option 28e in the Mendel Version 16 Manual [Section 28.1,  page 279](http://software.genetics.ucla.edu/download?file=202). It consists of simulated data where the two traits of interest have one contributing SNP and a sex effect.

We use the OpenMendel package [SnpArrays.jl](https://openmendel.github.io/SnpArrays.jl/latest/) to read in the PLINK formatted SNP data. In example 2b, we follow Mendel Option 28e with the simulation parameters for Trait1 and Trait2 in Ped28e.out as shown below.

In both examples, you can specify your own arbitrary fixed effect sizes, variance components and simulation parameters as desired.

In the $\mathbf{Generating}$ $\mathbf{Effect}$ $\mathbf{Sizes}$ Section of Example 2), we show how the user can generate effect sizes that depend on the minor allele frequencies from a function such as an exponential or chisquare. To aid the user when they wish to include a large number of loci in the model, we created a function that automatically writes out the mean components. 

$\mathbf{AT}$ $\mathbf{THE}$ $\mathbf{END}$ $\mathbf{of}$ $\mathbf{Example}$ $\mathbf{1}$, we demo how to $\mathbf{write}$ $\mathbf{the}$ $\mathbf{results}$ of each simulation to a file on the users own machine.

### Double check that you are using Julia version 1.0.3 or higher by checking the machine information


```julia
versioninfo()
```

    Julia Version 1.0.3
    Commit 099e826241 (2018-12-18 01:34 UTC)
    Platform Info:
      OS: macOS (x86_64-apple-darwin14.5.0)
      CPU: Intel(R) Core(TM) i7-7700HQ CPU @ 2.80GHz
      WORD_SIZE: 64
      LIBM: libopenlibm
      LLVM: libLLVM-6.0.0 (ORCJIT, skylake)


# Add any missing packages needed for this tutorial:

Note: For demonstration purposes, the generation of this Jupyter Notebook requires the use of the following registered packages: `DataFrames.jl`, `SnpArrays.jl`, `StatsModels.jl`, `Random.jl`, `DelimitedFiles.jl`, `StatsBase.jl`, and `StatsFuns.jl`. 

If it is your first time using these registered packages, you will first have to add the registered packages: DataFrames, SnpArrays, StatsModels, Random, LinearAlgebra, DelimitedFiles, Random, StatsBase by running the following code chunk in Julia's package manager:


pkg> add DataFrames
pkg> add SnpArrays
...
pkg> add StatsFuns

You can also use the package manager to add the `TraitSimulation.jl` package by running the following link: </br>


pkg> add "https://github.com/sarah-ji/TraitSimulation.jl"

Only after all of the necessary packages have been added, load them into your working environment with the `using` command:


```julia
using DataFrames, SnpArrays, StatsModels, Random, LinearAlgebra, DelimitedFiles, StatsBase, TraitSimulation, StatsFuns
```

# Reproducibility

For reproducibility, we set a random seed using the `Random.jl` package for each simulation using `Random.seed!(1234)`.  If you wish to end up with different data, you will need to comment out these commands or use another value in Random.seed!().


```julia
Random.seed!(1234);
```

# The notebook is organized as follows:

## Example 1: Generalized Linear Model
In this example we show how to generate single or multiple traits from GLM's with a genetic variant in the fixed effects, but no residual familial correlation.

$\text{vec}(Y) \sim \text{Nomral}(X B, \Sigma_1 \otimes V_1 + \cdots + \Sigma_m \otimes V_m),$

where $\otimes$ is the [Kronecker product](https://en.wikipedia.org/wiki/Kronecker_product).

In this model, **data** is represented by  

* `Y`: `n x d` response matrix  
* `X`: `n x p` covariate matrix  
* `V=(V1,...,Vm)`: a tuple `m` `n x n` covariance matrices  

and **parameters** are  

* `B`: `p x d` mean parameter matrix  
* `Σ=(Σ1,...,Σm)`: a tuple of `m` `d x d` variance components  

### (a) Single Trait:
#### $$Y_{1} ∼ N(\mu_{1}, \sigma^{2})$$

In example (1a) we simulate a $\textbf{SINGLE INDEPENDENT NORMAL TRAIT}$, with simulation parameters: $\mu_{1} = 20 + 3*sex - 1.5*locus$, $\sigma^{2} = 2$

### (b) Multiple Independent Traits:
 #### $$ Y = 
 \begin{bmatrix}
Y_{1}\\
Y_{2}
\end{bmatrix}, Y_{1} \!\perp\!\!\!\perp Y_{2}$$  &nbsp; &nbsp;  
$$ Y_{1} ∼ N(\mu_{1}, 2), \mu_{1} = 40 + 3(sex) - 1.5(locus), Y_{2} ∼ Poisson(\mu_{1}), \mu_{1} = 2 + 2(sex) - 1.5(locus))$$

In example (1b) we simulate $\textbf{TWO INDEPENDENT TRAITS SIMULTANEOUSLY}$, one from a Normal distribution and one from a Poisson distribution.<br>


## Example 2: Linear Mixed Model
In this example we show how to generate data with the additional additive genetic variance component or residual correlation among relatives. 

For convenience we use the common assumption that the residual covariance among two relatives can be captured by the additive genetic variance times twice the kinship coefficient. However, if you like you can specify your own variance components and their design matrices as long as they are positive semi definite using the `@vc` macro demonstrated in this example. 

### (a) Single Trait:

#### $$Y_{1} ∼ N(\mu_{1}, 4* 2GRM + 2I)$$

In example (1a) we simulate a $\textbf{SINGLE TRAIT CONTROLLING FOR FAMILY STRUCTURE}$, with the corresponding Mendel Example 28e Simulation parameters, location : $\mu_{1} = 40 + 3*sex - 1.5*locus$, scale : $V$ =  $V_{{a}_{1,1}}* 2GRM + V_{{e}_{1, 1}}I_{n} = 4* 2GRM + 2I$. 


### (b) Multiple Correlated Traits: (Mendel Example 28e Simulation)


$$ 
Y = 
\begin{bmatrix}
Y_{1}\\
Y_{2}
\end{bmatrix}, Y_{1} \not\!\perp\!\!\!\perp Y_{2}
$$


$$Y \sim N(\mathbf{\mu},\Sigma  = V_{a} \otimes (2GRM) + V_{e} \otimes I_{n}) ,$$


$$
\mathbf{\mu} = \begin{bmatrix}
\mu_1 \\
\mu_2 \\
\end{bmatrix}
= \begin{bmatrix}
40 + 3(sex) - 1.5(locus)\\
20 + 2(sex) - 1.5(locus)\\
\end{bmatrix} , V_{a} = \begin{bmatrix}
4 & 1\\
1 & 4\\
\end{bmatrix} , V_e 
= \begin{bmatrix}
2 & 0\\
0 & 2\\
\end{bmatrix}
$$


We simulate $\textbf{TWO CORRELATED TRAITS CONTROLLING FOR FAMILY STRUCTURE}$ with simulation parameters, location = $\mu$ and scale = $\Sigma$. 


## Example 3: Rare Variant Linear Mixed Model 

The example also assumes an additive genetic variance component in the model which includes 20 rare SNPs, defined as snps with minor allele frequencies greater than 0.002 but less than 0.02.  In practice rare SNPs have smaller minor allele frequencies, but we are limited in this tutorial by the number of individuals in the data set. <br>

We simulate a Single normal Trait controlling for family structure, with effect sizes generated as a function of the minor allele frequencies.

#### $$ Y_{2} ∼ N(\mu_{rare20}, 4* 2GRM + 2I)
$$

# Reading the Mendel 28a data using SnpArrays

First use `SnpArrays.jl` to read in the SNP data



```julia
snpdata = SnpArray("traitsim28e.bed", 212)
```




    212×253141 SnpArray:
     0x03  0x03  0x00  0x03  0x03  0x03  …  0x02  0x02  0x00  0x00  0x03  0x00
     0x03  0x03  0x00  0x02  0x02  0x03     0x00  0x03  0x00  0x00  0x03  0x00
     0x03  0x03  0x00  0x03  0x03  0x03     0x03  0x02  0x00  0x00  0x03  0x00
     0x03  0x03  0x00  0x03  0x03  0x03     0x02  0x03  0x00  0x00  0x03  0x00
     0x03  0x03  0x00  0x03  0x03  0x03     0x00  0x03  0x00  0x00  0x03  0x00
     0x03  0x03  0x00  0x03  0x03  0x03  …  0x00  0x00  0x00  0x00  0x00  0x03
     0x03  0x02  0x00  0x03  0x03  0x03     0x02  0x03  0x00  0x03  0x00  0x03
     0x03  0x03  0x00  0x03  0x03  0x03     0x02  0x03  0x00  0x03  0x00  0x03
     0x03  0x02  0x00  0x03  0x03  0x03     0x02  0x02  0x00  0x02  0x00  0x03
     0x03  0x02  0x00  0x03  0x03  0x03     0x03  0x03  0x00  0x03  0x00  0x03
     0x03  0x02  0x00  0x03  0x03  0x03  …  0x00  0x02  0x00  0x02  0x00  0x03
     0x03  0x03  0x00  0x03  0x03  0x03     0x00  0x02  0x00  0x02  0x00  0x03
     0x03  0x02  0x00  0x03  0x03  0x03     0x02  0x02  0x00  0x02  0x00  0x03
        ⋮                             ⋮  ⋱     ⋮                             ⋮
     0x03  0x03  0x00  0x03  0x03  0x03  …  0x00  0x03  0x00  0x00  0x03  0x00
     0x03  0x03  0x00  0x03  0x03  0x03     0x00  0x03  0x00  0x02  0x02  0x02
     0x03  0x03  0x00  0x03  0x03  0x03     0x00  0x02  0x00  0x00  0x03  0x00
     0x03  0x02  0x00  0x02  0x02  0x03     0x02  0x03  0x00  0x03  0x00  0x03
     0x03  0x03  0x00  0x02  0x02  0x03     0x02  0x03  0x00  0x00  0x03  0x00
     0x03  0x03  0x00  0x03  0x03  0x03  …  0x02  0x03  0x00  0x02  0x02  0x00
     0x03  0x03  0x00  0x02  0x02  0x03     0x02  0x03  0x00  0x00  0x02  0x02
     0x03  0x03  0x00  0x03  0x03  0x03     0x00  0x03  0x00  0x00  0x03  0x00
     0x03  0x02  0x00  0x03  0x03  0x03     0x02  0x03  0x00  0x00  0x02  0x02
     0x03  0x03  0x00  0x03  0x03  0x03     0x00  0x03  0x00  0x00  0x03  0x00
     0x03  0x03  0x00  0x03  0x03  0x03  …  0x02  0x03  0x00  0x00  0x03  0x00
     0x03  0x03  0x00  0x03  0x03  0x03     0x00  0x03  0x00  0x02  0x02  0x02



Store the FamID and PersonID of Individuals in Mendel 28e data


```julia
famfile = readdlm("traitsim28e.fam", ',')
Fam_Person_id = DataFrame(FamID = famfile[:, 1], PID = famfile[:, 2])
```




Note: Because later we will want to compare our results to the original results in the file,  we subset `traits_original` 


```julia
traits_original = DataFrame(Trait1 = famfile[:, 7], Trait2 = famfile[:, 8])
```

Transform sex variable from M/F to 1/-1 as does Mendel 28e data.  If you prefer you can use the more common convention of making one of the sexes the reference sex (coding it as zero) and make the other sex have the value 1.


```julia
sex = map(x -> strip(x) == "F" ? -1.0 : 1.0, famfile[:, 5]) # note julia's ternary operator '?'
```




    212-element Array{Float64,1}:
     -1.0
     -1.0
      1.0
      1.0
     -1.0
     -1.0
      1.0
      1.0
     -1.0
      1.0
     -1.0
      1.0
     -1.0
      ⋮  
      1.0
      1.0
      1.0
     -1.0
      1.0
      1.0
      1.0
      1.0
      1.0
      1.0
      1.0
      1.0



### Names of Variants:

We want to find the index of the causal snp, rs10412915, in the snp_definition file and then subset that snp from the genetic marker data above. 
We first subset the SNP names into a vector called `snpid`


```julia
snpdef28_1 = readdlm("traitsim28e.bim", Any; header = false)
snpid = map(x -> strip(string(x)), snpdef28_1[:, 1]) # strip mining in the data 
```




    253141-element Array{SubString{String},1}:
     "rs3020701"  
     "rs56343121" 
     "rs143501051"
     "rs56182540" 
     "rs7260412"  
     "rs11669393" 
     "rs181646587"
     "rs8106297"  
     "rs8106302"  
     "rs183568620"
     "rs186451972"
     "rs189699222"
     "rs182902214"
     ⋮            
     "rs188169422"
     "rs144587467"
     "rs139879509"
     "rs143250448"
     "rs145384750"
     "rs149215836"
     "rs139221927"
     "rs181848453"
     "rs138318162"
     "rs186913222"
     "rs141816674"
     "rs150801216"



We first need to find the position of the snp rs10412915.  If you wish to use another snp just change the rs number to another one that is found in the available genotype data, for example rs186913222.


```julia
ind_rs10412915 = findall(x -> x == "rs10412915", snpid)[1]
```




    236074



We see that the causal snp, rs10412915, is the 236074th variant in the snp dataset.

Let's create a design matrix for the alternative model that includes sex and locus rs10412915.


```julia
locus = convert(Vector{Float64}, @view(snpdata[:, ind_rs10412915]))
X = DataFrame(sex = sex, locus = locus)
```

# Example 1) Multiple Independent Traits: User specified distributions

Here I simulate two independent traits simultaneously, one from a Normal distribution and the other from a Poisson Distribution. 
We create the following 3 vectors to specify the simulation parameters of the two independent traits: 

 `dist_type_vector` , `link_type_vector` , `mean_formulas`

$$
Y_{1b_{1}} ∼ N(\mu_{1b}, 2),  \mu_{1b} = 40 + 3(sex) - 1.5(locus)\\
Y_{1b_{2}} ∼ Poisson(\mu_{2b}),  \mu_{2b} = 2 + 2(sex) - 1.5(locus)\\
$$


```julia
#for multiple glm traits from different distributions
dist_type_vector = [NormalResponse(4), PoissonResponse()]
link_type_vector = [IdentityLink(), LogLink()]

mean_formulas = ["40 + 3(sex) - 1.5(locus)", "2 + 2(sex) - 1.5(locus)"]

Multiple_GLM_traits_model_NOTIID = Multiple_GLMTraits(mean_formulas, X, dist_type_vector, link_type_vector)
Simulated_GLM_trait_NOTIID = simulate(Multiple_GLM_traits_model_NOTIID)
```

```julia
describe(Simulated_GLM_trait_NOTIID, stats = [:mean, :std, :min, :q25, :median, :q75, :max, :eltype])
```

## Saving Simulation Results to Local Machine

Write the newly simulated trait into a comma separated (csv) file for later use. Note that the user can specify the separator to '\t' for tab separated, or another separator of choice. 

Here we output the simulated traits and covariates for each of the 212 individuals, labeled by their pedigree ID and person ID.


```julia
Trait1_GLM = hcat(Fam_Person_id, Simulated_GLM_trait_NOTIID, X)
```




```julia
#cd("/Users") #change to home directory
CSV.write("Trait1_GLM.csv", Trait1_GLM)
```




    "Trait1_GLM.csv"



# Example 2: Linear Mixed Model (with additive genetic variance component).
Examples 2a and 2c simulate single traits, while Example 2b simulates two correlated traits.

We make note that the user can extend the model in Example 2b to include more than 2 variance components using the `@vc` macro.


## The Variance Covariance Matrix

Recall : $E(\mathbf{GRM}) = \Phi$
<br>
We use the [SnpArrays.jl](https://github.com/OpenMendel/SnpArrays.jl) package to find an estimate of the Kinship ($\Phi$), the Genetic Relationship Matrix (GRM). 

We will use the same values of $\mathbf{GRM}$, $V_a$, and $V_e$ for both the mixed effect example and for the rare variant example.

Note that the residual covariance among two relatives is the additive genetic variance, $V_a$, times twice the kinship coefficient, $\Phi$. The kinship matrix is derived from the genetic relationship matrix (GRM) across the common SNPs with minor allele frequency at least 0.05.


```julia
GRM = grm(snpdata, minmaf=0.05)
```




    212×212 Array{Float64,2}:
      0.498264     0.0080878    0.0164327   …   0.0246825    0.00181856
      0.0080878    0.498054    -0.0212599      -0.0285927   -0.0226525 
      0.0164327   -0.0212599    0.499442       -0.0219661   -0.00748536
      0.253627    -0.00160532   0.282542        0.00612693  -0.00339125
      0.126098     0.253365     0.128931       -0.0158446   -0.00633959
     -0.014971    -0.00266073  -0.00243384  …   0.00384757   0.0145936 
     -0.0221357    0.0100492   -0.0107012      -0.0148443   -0.00127783
     -0.01629     -0.00749253  -0.015372       -0.0163305   -0.00258392
     -0.016679     0.00353587  -0.0128844      -0.0332489   -0.00707839
     -0.0176101   -0.00996912  -0.0158473      -0.00675875  -0.0122339 
     -0.0162558    0.00938592   0.0064231   …  -0.00510882   0.0168778 
     -0.0167487    0.00414544  -0.00936538     -0.0134863    0.0020952 
     -0.031148     0.00112387  -0.010794        0.00383105   0.0198635 
      ⋮                                     ⋱   ⋮                      
     -0.00865735  -0.00335548  -0.0148433   …   0.00806601  -0.0211537 
      0.00296028   0.0043655   -0.0183683       0.0012496    0.00898193
     -0.0204601   -0.0270898   -0.00194048     -0.0185883   -0.0116621 
     -0.0174561   -0.0128509   -0.0155773      -0.0274183   -0.0063823 
     -0.00170995   0.0154211   -0.00168146     -0.00684865  -0.0067438 
      0.00718047  -0.00525265  -0.00283975  …   0.0309601    0.0261103 
     -0.0170218   -0.00661916   0.0020924      -0.022858     0.0037451 
      0.0142551    0.0208073    0.0096287       0.00598877   0.0094809 
     -0.00586031  -0.00733706   0.0339257       0.0109116   -0.0177771 
      0.00299024  -0.0134027    0.0150825       0.00799507   0.0150077 
      0.0246825   -0.0285927   -0.0219661   …   0.593999     0.0497083 
      0.00181856  -0.0226525   -0.00748536      0.0497083    0.491743  




```julia
V_A = [4 1; 1 4]
V_E = [2.0 0.0; 0.0 2.0]
I_n = Matrix{Float64}(I, size(GRM));
```

### Example 2a: Single Trait 
$$
Y_{2a} ∼ N(μ_1, 4* 2GRM + 2I)$$

We simulate a Normal Trait controlling for family structure, location = $\mu_1$ and scale =  $\mathbf{V} = 2*V_a \Phi + V_e I = 4* 2GRM + 2I$. 



```julia
mean_formula = ["40 + 3(sex) - 1.5(locus)"]
```




    1-element Array{String,1}:
     "40 + 3(sex) - 1.5(locus)"




```julia
Ex2a_model = LMMTrait(mean_formula, X, 4*(2*GRM) + 2*(I_n))
trait_2a = simulate(Ex2a_model)
```


```julia
describe(trait_2a, stats = [:mean, :std, :min, :q25, :median, :q75, :max, :eltype])
```


##  Example 2b) Multiple Correlated Traits: (Mendel Example 28e Simulation)

We simulate two correlated Normal Traits controlling for family structure, location = $μ$ and scale = $\mathbf\Sigma$.
The corresponding variance covariance matrix as specified Mendel Option 28e, $\mathbf{Σ}$, is generated here.

$$
Y_{2b} ∼ N(μ, \mathbf\Sigma)
$$ 

$$
\mathbf{\mu} = \begin{vmatrix}
\mu_1 \\
\mu_2 \\
\end{vmatrix}
= \begin{vmatrix}
40 + 3(sex) - 1.5(locus)\\
20 + 2(sex) - 1.5(locus)\\
\end{vmatrix}
\\
$$

$$
\mathbf\Sigma  = V_a \otimes (2GRM) + V_e \otimes I_n
$$


&nbsp; $FYI$: To create a trait with different variance components change the elements of $\mathbf\Sigma$. We create the variance component object `variance_formula` below, to simulate our traits in example 2b. While this tutorial only uses 2 variance components, we make note that the `@vc` macro is designed to handle as many variance components as needed. 

As long as each Variance Component is specified correctly, we can create a `VarianceComponent` Julia object for Trait Simulation:

&nbsp; 
Example) Specifying more than 2 variance components (let V_B indicate an additional Environmental Variance component) 
```{julia}
multiple_variance_formula = @vc V_A ⊗ GRM + V_E1 ⊗ I_n + V_E2 ⊗ I_n + V_E3 ⊗ I_n;
```



```julia
# @vc is a macro that creates a 'VarianceComponent' Type for simulation
variance_formula = @vc V_A ⊗ GRM + V_E ⊗ I_n;
```

These are the formulas for the fixed effects, as specified by Mendel Option 28e.


```julia
mean_formulas = ["40 + 3(sex) - 1.5(locus)", "20 + 2(sex) - 1.5(locus)"]
```




    2-element Array{String,1}:
     "40 + 3(sex) - 1.5(locus)"
     "20 + 2(sex) - 1.5(locus)"




```julia
Ex2b_model = LMMTrait(mean_formulas, X, variance_formula)
trait_2b = simulate(Ex2b_model)
```


### Summary Statistics of Our Simulated Traits


```julia
describe(trait_2b, stats = [:mean, :std, :min, :max, :eltype])
```


### Summary Statistics of the Original Mendel 28e dataset Traits:

Note we want to see similar values from our simulated traits!


```julia
describe(traits_original, stats = [:mean, :std, :min, :max, :eltype])
```


## Example 2c) Rare Variant Linear Mixed Model with effect sizes as a function of the allele frequencies. 


$$
Y_{2c} ∼ N(\mu_{rare20}, 4* 2GRM + 2I)
$$

In this example we first subset only the rare SNP's with minor allele frequency greater than 0.002 but less than 0.02, then we simulate traits on 20 of the rare SNP's as fixed effects. For this demo, the indexing `snpid[rare_index][1:2:40]` allows us to subset every other rare snp in the first 40 SNPs, to get our list of 20 rare SNPs. Change the range and number of SNPs to simulate with more or less SNPs and from different regions of the genome. The number 20 is arbitrary and you can use more or less than 20 if you desire by changing the final number. You can change the spacing of the snps by changing the second number. 
For example, `snpid[rare_index][1:5:500]` would give you 100 snps.

Here are the 20 SNP's that will be used for trait simulation in this example.  



```julia
# filter out rare SNPS, get a subset of uncommon SNPs with 0.002 < MAF ≤ 0.02
minor_allele_frequency = maf(snpdata)
rare_index = (0.002 .< minor_allele_frequency .≤ 0.02)
data_rare = snpdata[:, rare_index];
```


```julia
maf_20_rare_snps = minor_allele_frequency[rare_index][1:2:40]
rare_snps_for_simulation = snpid[rare_index][1:2:40]
```




    20-element Array{SubString{String},1}:
     "rs3020701"  
     "rs181646587"
     "rs182902214"
     "rs184527030"
     "rs10409990" 
     "rs185166611"
     "rs181637538"
     "rs186213888"
     "rs184010370"
     "rs11667161" 
     "rs188819713"
     "rs182378235"
     "rs146361744"
     "rs190575937"
     "rs149949827"
     "rs117671630"
     "rs149171388"
     "rs188520640"
     "rs142722885"
     "rs146938393"



## Generating Effect Sizes

In practice rare SNPs have smaller minor allele frequencies but we are limited in this tutorial by the number of individuals in the data set. We use generated effect sizes to evaluate $\mu_{rare20}$ on the following Dataframe: <br> 


```julia
geno_rare20_converted = convert(DataFrame, convert(Matrix{Float64}, @view(data_rare[:, 1:2:40])))
names!(geno_rare20_converted, Symbol.(rare_snps_for_simulation))
```


### Chisquared Distribution (df = 1)

## Generating Effect Sizes Based on MAF

For demonstration purposes, we simulate effect sizes from the Chi-squared(df = 1) distribution, where we use the minor allele frequency (maf) as x and find f(x) where f is the pdf for the Chi-squared (df = 1) density, so that the rarest SNP's have the biggest effect sizes. The effect sizes are rounded to the second digit, throughout this example. Notice there is a random +1 or -1, so that there are effects that both increase and decrease the simulated trait value.

In addition to the Chi-Squared distribution, we also demo how to simulate from the Exponential distribution, where we use the minor allele frequency (maf) as x and find f(x) where f is the pdf for the Exponential density.


```julia
# Generating Effect Sizes from Chisquared(df = 1) density
n = length(maf_20_rare_snps)
chisq_coeff = zeros(n)

for i in 1:n
    chisq_coeff[i] = sign(rand() - .5) * chisqpdf(1, maf_20_rare_snps[i])/5.0
end
```

Take a look at the simulated coefficients on the left, next to the corresponding minor allele frequency. Notice the rarer SNPs have the largest absolute values for their effect sizes.


```julia
Ex2c_rare = round.([chisq_coeff maf_20_rare_snps], digits = 3)
Ex2c_rare = DataFrame(Chisq_Coefficient = Ex2c_rare[:, 1] , MAF_rare = Ex2c_rare[:, 2] )
```



```julia
simulated_effectsizes_chisq = Ex2c_rare[:, 1]
```




    20-element Array{Float64,1}:
     -0.616
     -0.666
     -0.818
     -0.575
      0.818
      1.159
     -0.945
     -0.818
      0.945
     -1.641
     -0.666
      1.159
      1.641
     -1.159
     -0.575
      1.641
      1.641
     -1.641
      0.575
     -1.159



### Exponential Distribution
Here we show how to generate effect sizes for the 20 rare snp's from the Exponential Distribution, where we use the maf as x and find f(x) where f is the pdf for the Exponential density


```julia
simulated_effectsizes_exp = round.(6*exp.(-200*maf_20_rare_snps), digits = 3)
```




    20-element Array{Float64,1}:
     0.221
     0.354
     0.909
     0.138
     0.909
     2.336
     1.457
     0.909
     1.457
     3.744
     0.354
     2.336
     3.744
     2.336
     0.138
     3.744
     3.744
     3.744
     0.138
     2.336



## Function for Mean Model Expression

In some cases a large number of variants may be used for simulation. Thus, in this example we create a function where the user inputs a vector of coefficients and a vector of variants for simulation, then the function outputs the mean model expression. 

The function `FixedEffectTerms`, creates the proper evaluated expression for the simulation process, using the specified vectors of coefficients and snp names. The function outputs `evaluated_fixed_expression` which will be used to estimate the mean effect, `μ` in our mixed effects model. We make use of this function in this example, instead of having to write out all 20 of the coefficients and variant locus names.


```julia
rare_snps_for_simulation
```




    20-element Array{SubString{String},1}:
     "rs3020701"  
     "rs181646587"
     "rs182902214"
     "rs184527030"
     "rs10409990" 
     "rs185166611"
     "rs181637538"
     "rs186213888"
     "rs184010370"
     "rs11667161" 
     "rs188819713"
     "rs182378235"
     "rs146361744"
     "rs190575937"
     "rs149949827"
     "rs117671630"
     "rs149171388"
     "rs188520640"
     "rs142722885"
     "rs146938393"




```julia
function FixedEffectTerms(effectsizes::AbstractVecOrMat, snps::AbstractVecOrMat)
 # implementation
    fixed_terms = ""
for i in 1:length(simulated_effectsizes_chisq) - 1
expression = " + " * string(simulated_effectsizes_chisq[i]) * "(" * rare_snps_for_simulation[i] * ")"
    fixed_terms = fixed_terms * expression
end
    return String(fixed_terms)
end

```




    FixedEffectTerms (generic function with 1 method)




```julia
mean_formula_rare = FixedEffectTerms(simulated_effectsizes_chisq, rare_snps_for_simulation)
```




    " + -0.616(rs3020701) + -0.666(rs181646587) + -0.818(rs182902214) + -0.575(rs184527030) + 0.818(rs10409990) + 1.159(rs185166611) + -0.945(rs181637538) + -0.818(rs186213888) + 0.945(rs184010370) + -1.641(rs11667161) + -0.666(rs188819713) + 1.159(rs182378235) + 1.641(rs146361744) + -1.159(rs190575937) + -0.575(rs149949827) + 1.641(rs117671630) + 1.641(rs149171388) + -1.641(rs188520640) + 0.575(rs142722885)"



## Example 2c) Mixed effects model Single Trait and rare variants:
$$
Y_{2c} ∼ N(μ_{20raresnps}, 4* 2GRM + 2I)$$



```julia
rare_20_snp_model = LMMTrait([mean_formula_rare], geno_rare20_converted, 4*(2*GRM) + 2*(I_n))
trait_rare_20_snps = simulate(rare_20_snp_model)
```


```julia
describe(trait_rare_20_snps, stats = [:mean, :std, :min, :max, :eltype])
```


## Saving Simulation Results to Local Machine

Here we output the simulated trait values for each of the 212 individuals, labeled by their pedigree ID and person ID.

In addition, we output the genotypes for the variants used to simulate this trait.


```julia
Trait2_mixed = hcat(Fam_Person_id, trait_rare_20_snps, geno_rare20_converted)
```




```julia
Coefficients = DataFrame(Coefficients = simulated_effectsizes_chisq)
SNPs_rare = DataFrame(SNPs = rare_snps_for_simulation)
Trait2_mixed_sim = hcat(Coefficients, SNPs_rare)
```


```julia
#cd("/Users") #change to home directory
using CSV
CSV.write("Trait2c_mixed.csv", Trait2_mixed)
CSV.write("Trait2c_mixed_sim.csv", Trait2_mixed_sim);
```

## Citations: 

[1] Lange K, Papp JC, Sinsheimer JS, Sripracha R, Zhou H, Sobel EM (2013) Mendel: The Swiss army knife of genetic analysis programs. Bioinformatics 29:1568-1570.`


[2] OPENMENDEL: a cooperative programming project for statistical genetics.
[Hum Genet. 2019 Mar 26. doi: 10.1007/s00439-019-02001-z](https://www.ncbi.nlm.nih.gov/pubmed/?term=OPENMENDEL).

