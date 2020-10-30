using SnpArrays, Test, Statistics, Random, TraitSimulation
using DataFrames, LinearAlgebra, OrdinalMultinomialModels, GLM, DelimitedFiles
Random.seed!(1234)

# demo genetic data can be found in Snparrays package directory
filepath = SnpArrays.datadir()
filename = "EUR_subset"
EUR = SnpArray(SnpArrays.datadir(filename * ".bed"));

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

n_sim = 10

writepheno(filename, filepath, trait_ordered, n_sim)

pheno_ordered = readdlm("pheno.txt", Any)

# test to see if the outputted phenotype file has the correct number of columns, the first two columns are family ID and Individual ID respectively. The remaining columns are the # of simulated univariate traits
@test size(pheno_ordered, 2) == n_sim + 2 
# first column FID
@test famfile[:, 1] == string.(pheno_ordered[2:end, 1])
# second column IID 
@test famfile[:, 2] == string.(pheno_ordered[2:end, 2])


## VCM trait bivariate test
β = [β_full β_full]
GRM = grm(EUR, minmaf = 0.05);
I_n = Matrix(I, size(GRM))
ΣE = [1.0 0.0; 0.0 1.0]
ΣA = [0.3 0.0; 0.0 0.3]
vc = @vc ΣA ⊗ 2GRM + ΣE ⊗ I_n;
trait_vcm = VCMTrait(X, β, vc)

n_sim = 5

writepheno(filename, filepath, trait_vcm, n_sim)

pheno_ordered = readdlm("pheno.txt", Any)

# test to see if the outputted phenotype file has the correct number of columns, the first two columns are family ID and Individual ID respectively. The remaining columns are the # of simulated univariate traits
n_traits = ntraits(trait_vcm)

@test size(pheno_ordered, 2) == n_traits * n_sim + 2 
# first column FID
@test famfile[:, 1] == string.(pheno_ordered[2:end, 1])
# second column IID 
@test famfile[:, 2] == string.(pheno_ordered[2:end, 2])
