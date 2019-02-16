using Distributions
using DataFrames
using LinearAlgebra
using TraitSimulation

#in context of linear mixed models and multiple correlated traits, the outcome must be multivariate normal with 
#GRM and environmental variance matrix or constant (iid) B
 
function multiple_trait_simulation(formulas, dataframe, B, GRM)
# for multiple traits
#find the number of traits
n_traits = length(formulas)

mean = Vector{Vector{Float64}}(undef, n_traits)
#for each trait
for i in 1:n_traits
	#calculate the mean vector
	mean[i] = mean_formula(formulas[i], dataframe)
end

#concatenate them together
meanvector = vcat(mean...) # take all of the i's and splat them into the meanvector.

A = cov(hcat(mean...)) # me assuming that we dont know this A and I have to assume it internally
term1 = kron(A, GRM) 
term2 = kron(B, Matrix{Float64}(I, size(GRM)))
Σ = term1 + term2

model = MvNormal(meanvector, Σ)
out1 = rand(model)

out2 = DataFrame(reshape(out1, (size(GRM)[1], n_traits)))

out2 = names!(out2, [Symbol("trait$i") for i in 1:n_traits])

return out2

end 

#test 
# cd /Users/sarahji/Desktop/_MendelBase_updated_for_v0.7 
# snps = SnpArray("heritability.bed")
# minor_allele_frequency = maf(snps)
# common_snps_index = (0.05 .≤ minor_allele_frequency)
# common_snps = SnpArrays.filter("heritability", trues(212), common_snps_index)
# df = convert(Matrix{Float64}, snps)
#df = DataFrame(df)
# GRM = grm(common_snps)
# B = [0.3 0.1 0.01; 0.1 0.3 0.1; 0.01 0.1 0.3]
# formulaszzz = ["1 + 3(x1)", "1 + 3(x2) + abs(x3)", "1 + x2*x3 + 0.2x5"]
# multiple_trait_simulation(formulaszzz, df, B, GRM)