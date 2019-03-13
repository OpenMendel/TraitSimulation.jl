using DataFrames, Distributions, SnpArrays, StatsModels, Random, LinearAlgebra, TraitSimulation

#cd /Users/sarahji/Desktop/OpenMendel_Sarah/Tutorials/Heritability 

# 1. DATA IMPORT:
#reads the bed file of snp data and then turns into named dataframe for simulation
;cd /Users/sarahji/Desktop/OpenMendel_Sarah/Tutorials/Heritability 
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

#for multiple glm traits from different distributions
dist_type_vector = [dist_N01, dist_N05]

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


#### TESTING 
# B = 2000
# variancecomp = @vc A_1 ⊗ B_1 + A_2 ⊗ B_2


# function testing(trait::LMMTrait)
# 	Y = Vector{Matrix{Float64}}(undef, 2000)
# 	ybar = zeros(B, length(trait.formulas))
# 	diff = zeros()
# for i in 1:B
# 	for j in 1:length(trait.formulas)
# 	Y[i] = multiple_trait_simulation7(zeros(212, 2), variancecomp)
# 	ybar[i, j] = mean(Y[i][:, j])
# 	diff = (Y[i][:, j] - ybar[i, j])
# end
# sample_variance = sum(abs2, diff)
# end

# ybar = mean(Y)
# sample_variance = sum(abs2, Y - ybar)


