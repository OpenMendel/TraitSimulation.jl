## Example 3: Rare Variant Linear Mixed Model 

This example is meant to simulate data in a scenario in which a number of rare mutations in a single gene can change a trait value.  In this example we model the residual variation among relatives with the additive genetic variance component and we include 20 rare variants in the mean portion of the model, defined as loci with minor allele frequencies greater than 0.002 but less than 0.02.  In practice rare variants have smaller minor allele frequencies, but we are limited in this tutorial by the relatively small size of the data set. Note also that our modeling these effects as part of the mean is not meant to imply that the best way to detect them would be a standard association analysis. Instead we recommend a burden or SKAT test. 

Specifically we are generating a single normal trait controlling for family structure with residual heritability of 67 percent, and effect sizes for the variants generated as a function of the minor allele frequencies. The rarer the variant the greater its effect size.

We run this simulation 1000 times, and store the simulation results in a vector of DataFrames. At the end of this example we write the results of the first of the 1000 replicates to a file on your own machine.

$$Y \sim N(\mathbf{\mu},\Sigma  = 4 x (2GRM) + 2 x I_{n}, nreps = 1000) ,$$

$\text{vec}(Y) \sim \text{Normal}(X B, \Sigma_1 \otimes V_1 + \cdots + \Sigma_m \otimes V_m),$

In this example we first subset only the rare SNP's with minor allele frequency greater than 0.002 but less than 0.02, then we simulate traits on 20 of the rare SNP's as fixed effects. For this demo, the indexing `snpid[rare_index][1:2:40]` allows us to subset every other rare snp in the first 40 SNPs, to get our list of 20 rare SNPs. Change the range and number of SNPs to simulate with more or less SNPs and from different regions of the genome. The number 20 is arbitrary and you can use more or less than 20 if you desire by changing the final number. You can change the spacing of the snps by changing the second number. 
For example, `snpid[rare_index][1:5:500]` would give you 100 snps.


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



## Generating Effect Sizes (Based on MAF)

In practice rare SNPs have smaller minor allele frequencies but we are limited in this tutorial by the number of individuals in the data set. We use generated effect sizes to evaluate $\mu_{rare20}$ on the following Dataframe: <br> 

### Chisquared Distribution (df = 1)

For demonstration purposes, we simulate effect sizes from the Chi-squared(df = 1) distribution, where we use the minor allele frequency (maf) as x and find f(x) where f is the pdf for the Chi-squared (df = 1) density, so that the rarest SNP's have the biggest effect sizes. The effect sizes are rounded to the second digit, throughout this example. Notice there is a random +1 or -1, so that there are effects that both increase and decrease the simulated trait value.


```julia
# Generating Effect Sizes from Chisquared(df = 1) density
n = length(maf_20_rare_snps)
chisq_coeff = zeros(n)

for i in 1:n
    chisq_coeff[i] = rand([-1, 1]) .* (0.1 / sqrt.(maf_20_rare_snps[i] .* (1 - maf_20_rare_snps[i])))
end
```

Take a look at the simulated coefficients on the left, next to the corresponding minor allele frequency. Notice the rarer SNPs have the largest absolute values for their effect sizes.


```julia
Ex3_rare = round.([chisq_coeff maf_20_rare_snps], digits = 3)
Ex3_rare = DataFrame(Chisq_Coefficient = Ex3_rare[:, 1] , MAF_rare = Ex3_rare[:, 2] )
```

```julia
simulated_effectsizes_chisq = Ex3_rare[:, 1]
```

    20-element Array{Float64,1}:
     -0.785
     -0.847
      1.034
     -0.735
      1.034
     -1.459
      1.193
     -1.034
     -1.193
      2.062
      0.847
      1.459
      2.062
     -1.459
      0.735
      2.062
      2.062
     -2.062
      0.735
     -1.459


## Function for Mean Model Expression

In some cases a large number of variants may be used for simulation. Thus, in this example we create a function where the user inputs a vector of coefficients and a vector of variants for simulation, then the function outputs the mean model expression. 

The function `FixedEffectTerms`, creates the proper evaluated expression for the simulation process, using the specified vectors of coefficients and snp names. The function outputs `evaluated_fixed_expression` which will be used to estimate the mean effect, in our mixed effects model. We make use of this function in this example, instead of having to write out all 20 of the coefficients and variant locus names.


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
