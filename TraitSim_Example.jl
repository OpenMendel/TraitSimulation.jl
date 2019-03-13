using DataFrames, Distributions, SnpArrays, StatsModels, Random, LinearAlgebra, TraitSimulation

#cd /Users/sarahji/Desktop/OpenMendel_Sarah/Tutorials/Heritability 

# 1. DATA IMPORT:
#reads the bed file of snp data and then turns into named dataframe for simulation
cd("/Users/sarahji/Desktop/OpenMendel_Sarah/Tutorials/Heritability")
snps = SnpArray("heritability.bed")
minor_allele_frequency = maf(snps)
common_snps_index = (0.05 .≤ minor_allele_frequency)
common_snps = SnpArrays.filter("heritability", trues(212), common_snps_index)
df = convert(Matrix{Float64}, snps)
df = DataFrame(df)

formulas = ["1 + 3(x1)", "1 + 3(x2) + abs(x3)"]

# Variance Specification for VCM: ex) @vc A ⊗ GRM + B ⊗ I
GRM = grm(common_snps)
A_1 = [0.3 0.1; 0.1 0.3]
B_1 = GRM
A_2 = [0.7 0.0; 0.0 0.7]
B_2 = Matrix{Float64}(I, size(GRM))

#Standard normal
dist_N01 = ResponseType(Normal(), IdentityLink(), 0.0, 1.0, 0.0, 0.0, 0)

#Normal(0, 5)
dist_N05 = ResponseType(Normal(), IdentityLink(), 0.0, 5.0, 0.0, 0.0, 0)

dist_Poisson5 = ResponseType(Poisson(), LogLink(), 5.0, 0.0, 0.0, 0.0, 0)

#for multiple glm traits from different distributions
dist_type_vector = [dist_N01, dist_Poisson5]

#SINGLE GLM TRAIT
GLM_trait_model = GLMTrait(formulas[1], df, dist_N01)
Simulated_GLM_trait = simulate(GLM_trait_model)


#MULTIPLE GLM TRAITS IID 
Multiple_iid_GLM_traits_model = Multiple_GLMTraits(formulas, df, dist_N01)
Simulated_GLM_trait = simulate(Multiple_iid_GLM_traits_model)

#MULTIPLE GLM TRAITS FROM DIFFERENT DISTRIBUTIONS

Multiple_GLM_traits_model_NOTIID = Multiple_GLMTraits(formulas, df, dist_type_vector)

Simulated_GLM_trait = simulate(Multiple_GLM_traits_model_NOTIID)

#LMM TRAIT
LMM_trait_model = LMMTrait(formulas, df, @vc A_1 ⊗ B_1 + A_2 ⊗ B_2)
Simulated_LMM_trait = simulate(LMM_trait_model)


### TESTING 

B = 2000
variancecomp = @vc A_1 ⊗ B_1 + A_2 ⊗ B_2

function testing(simulatedtrait::Matrix)
	n_people = size(simulatedtrait)[1]
	n_traits = size(simulatedtrait)[2]

	Y = Vector{Matrix{Float64}}(undef, B)
	ybar = Vector{Vector{Float64}}(undef, B)
	diff = Vector{Matrix{Float64}}(undef, B)
	sumabs2 = Vector{Float64}(undef, B)
for i in 1:B
	Y[i] = TraitSimulation.multiple_trait_simulation7(zeros(212, 2), variancecomp)
	ybar[i] = vec(mean(Y[i], dims = 1))
	diff[i] = Y[i] - reshape(repeat(ybar[i], inner = n_people), n_people, n_traits)
	sumabs2[i] = sum(abs2, sum(diff[i], dims = 1))
end

sample_mean = mean(ybar)
sample_variance = sum(sumabs2)/(B - 1)
return(sample_mean, sample_variance)
end

#testing(Matrix(Simulated_GLM_trait))


