
# Trait Simulation Tutorial


Authors: Sarah Ji, Janet Sinsheimer, Kenneth Lange

In this notebook we show how to use the `TraitSimulation.jl` package to simulate traits from genotype data from unrelateds or families with user-specified Generalized Linear Models (GLMs) or Linear Mixed Models (LMMs), respectively. For simulating under either GLM or LMMs, the user can specify the number of repitions for each simulation model. By default, the simulation will return the result of a single simulation. 

The data we will be using are from the Mendel version 16[1] sample files. The data are described in examples under Option 28e in the Mendel Version 16 Manual [Section 28.1,  page 279](http://software.genetics.ucla.edu/download?file=202). It consists of simulated data where the two traits of interest have one contributing SNP and a sex effect.

We use the OpenMendel package [SnpArrays.jl](https://openmendel.github.io/SnpArrays.jl/latest/) to read in the PLINK formatted SNP data. In example 1, we simulate generalized linear models assuming that everyone is unrelated. So in example 1,  the only data used from option 28e is the genotype for a specific locus in the snp file and the sex of the individual. The pedigree structure and relationship matrix are irrelevant.   In example 2 we simulate data under a linear mixed model so that we can model residual dependency among individuals.  In example 2b, we use the same parameters as were used in Mendel Option 28e with the simulation parameters for Trait1 and Trait2 in Ped28e.out as shown below.

In both examples, you can specify your own arbitrary fixed effect sizes, variance components and simulation parameters as desired. You can also specify the number of replicates for each Trait simulation in the `simulate` function.

In the $\mathbf{Generating}$ $\mathbf{Effect}$ $\mathbf{Sizes}$ Section of Example 2), we show how the user can generate effect sizes that depend on the minor allele frequencies from the chisquare distribution. To aid the user when they wish to include a large number of loci in the model, we created a function that automatically writes out the mean components for simulation.

$\textbf{At the end of Examples 1 and 3}$, we demo how to $\textbf{write the results}$ of the simulation to a file on your own machine.

### Double check that you are using Julia version 1.0 or higher by checking the machine information


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

# Add any missing packages needed for this tutorial:

If it is your first time using these registered packages, you will first have to add the registered packages: DataFrames, SnpArrays, StatsModels, Random, LinearAlgebra, DelimitedFiles, Random, StatsBase by running the following code chunk in Julia's package manager:

```{julia}
pkg> add DataFrames
pkg> add SnpArrays
pkg> add LinearAlgebra
pkg> add Random
```
You can also use the package manager to add the `TraitSimulation.jl` package by running the following link: </br>

```{julia}
pkg> add "https://github.com/sarah-ji/TraitSimulation.jl"
```

Only after all of the necessary packages have been added, load them into your working environment with the `using` command:


# The notebook is organized as follows:

## Example 1: Linear Mixed Model
In this example we show how to generate data so that the related individuals have correlated trait values even after we account for the effect of a snp, a combination of snps or other fixed effects.

For convenience we use the common assumption that the residual covariance among two relatives can be captured by the additive genetic variance times twice the kinship coefficient. However, if you like you can specify your own variance components and their design matrices as long as they are positive semi definite using the `@vc` macro demonstrated in this example. We run this simulation 1000 times, and store the simulation results in a vector of DataFrames.

### Multiple Correlated Traits: (Mendel Example 28e Simulation)

$$ Y = 
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


We simulate $\textbf{TWO CORRELATED TRAITS CONTROLLING FOR FAMILY STRUCTURE}$ with simulation parameters, location = $\mu$ and scale = $\Sigma$. By default, without specifying a number of replicates for the user specified LMM (like this example), the `simulate` function returns a single set of simulated traits.


## Example 2: Rare Variant Linear Mixed Model 

This example is meant to simulate data in a scenario in which a number of rare mutations in a single gene can change a trait value.  In this example we model the residual variation among relatives with the additive genetic variance component and we include 20 rare variants in the mean portion of the model, defined as loci with minor allele frequencies greater than 0.002 but less than 0.02.  In practice rare variants have smaller minor allele frequencies, but we are limited in this tutorial by the relatively small size of the data set. Note also that our modeling these effects as part of the mean is not meant to imply that the best way to detect them would be a standard association analysis. Instead we recommend a burden or SKAT test.

Specifically we are generating a single normal trait controlling for family structure with residual heritabiity of 67%, and effect sizes for the variants generated as a function of the minor allele frequencies. The rarer the variant the greater its effect size.

$$ Y \sim N(\mu_, 4 * 2GRM + 2I)
$$


## Citations: 

[1] Lange K, Papp JC, Sinsheimer JS, Sripracha R, Zhou H, Sobel EM (2013) Mendel: The Swiss army knife of genetic analysis programs. Bioinformatics 29:1568-1570.`


[2] OPENMENDEL: a cooperative programming project for statistical genetics.
[Hum Genet. 2019 Mar 26. doi: 10.1007/s00439-019-02001-z](https://www.ncbi.nlm.nih.gov/pubmed/?term=OPENMENDEL).

