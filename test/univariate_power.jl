using SnpArrays, Test, Statistics, Random, TraitSimulation
using DataFrames, LinearAlgebra, OrdinalMultinomialModels, GLM
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

# first ordinal power for the single snp
θ = [1.0, 1.2, 1.4]
trait_ordered = OrderedMultinomialTrait(X, β_full, θ, LogitLink())

nsim = 10
γs = collect(0.0:0.05:0.5)

randomseed = 12345
ordinal_pvals = power_simulation(nsim, γs, trait_ordered, randomseed)

alpha = 0.005
powers = power(ordinal_pvals, alpha)
@test powers[1] < powers[end]


# next glm power
dist = Poisson()
link = LogLink()
pdf_age = Normal(45, 8)
# simulate age under the specified pdf_age and standardize to be ~ N(0, 1)
age = rand(pdf_age, length(sex))
X = [sex age locus]
β_full =
[2.52
1.0
0.05]
glm_trait = GLMTrait(X, β_full, dist, link)

nsim = 10
γs = collect(0.0:0.5:2.0)

randomseed = 12345
glm_pvals = power_simulation(nsim, γs, glm_trait, randomseed)

alpha = 0.20
powers = power(glm_pvals, alpha)
@test powers[1] < powers[end]
