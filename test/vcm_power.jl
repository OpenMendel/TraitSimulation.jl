using SnpArrays, Distributions, Test, Statistics, VarianceComponentModels, Distributions, Random
using DataFrames, LinearAlgebra, StatsBase, TraitSimulation, Distributions
Random.seed!(1234)

# demo genetic data can be found in Snparrays package directory
filename = "EUR_subset"
EUR = SnpArray(SnpArrays.datadir(filename * ".bed"));
GRM = grm(EUR, minmaf = 0.05);
EUR_data = SnpData(SnpArrays.datadir(filename));
bimfile = EUR_data.snp_info # store the snp_info with the snp names
snpid  = bimfile[:, :snpid] # store the snp names in the snpid vector
causal_snp_index = findfirst(x -> x ==  "rs62057050" , snpid) # find the index of the snp of interest by snpid

locus = convert(Vector{Float64}, @view EUR[:, causal_snp_index])
famfile = EUR_data.person_info
sex = map(x -> strip(x) == "F" ? 0.0 : 1.0, famfile[!, :sex])
X = [sex locus]
β_full =
 [0.52
0.05]

I_n = Matrix(I, size(GRM))
vc = @vc [0.01][:,:] ⊗ (GRM + I_n) + [0.9][:, :] ⊗ I_n
trait = VCMTrait(X, β_full[:, :], vc)
X = trait.X
V1 = trait.vc[1].V
V2 = trait.vc[2].V

nsim = 5
γs = collect(0.0:0.05:0.4)
Y = zeros(size(trait.μ))
X = trait.X
V1 = trait.vc[1].V
V2 = trait.vc[2].V
@test eltype(Y) == Float64

# allocate variance component objects for null and alternative hypothesis
vcm_null, vcrot_null, vcm_alt, vcrot_alt = @time null_and_alternative_vcm_and_rotate(Y, X, V1, V2)
pvalue = power_simulation(trait, γs, nsim, vcm_null, vcrot_null, vcm_alt, vcrot_alt)

@test size(pvalue) == (nsim, length(γs))

alpha = 0.0000005
powers = power(pvalue, alpha)
@test powers[1] < powers[end]
