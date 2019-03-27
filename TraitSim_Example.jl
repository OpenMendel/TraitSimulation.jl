using DataFrames, SnpArrays, StatsModels, Random, LinearAlgebra, TraitSimulation

#cd /Users/sarahji/Desktop/OpenMendel_Sarah/Tutorials/Heritability 

# 1. DATA IMPORT:
#reads the bed file of snp data and then turns into named dataframe for simulation
cd("/Users/sarahji/Desktop/OpenMendel_Sarah/Tutorials/Heritability")
snps = SnpArray("heritability.bed")
minor_allele_frequency = maf(snps)
common_snps_index = (0.05 .≤ minor_allele_frequency)
common_snps = SnpArrays.filter("heritability", trues(212), common_snps_index)
df = convert(Matrix{Float64}, @view(snps[:, :]))
df = DataFrame(df)

formulas = ["1 + 5(x1)", "1 + abs(sin(x2))"]

# Variance Specification for VCM: ex) @vc A ⊗ GRM + B ⊗ I
GRM = grm(common_snps)
A_1 = [0.3 0.1; 0.1 0.3]
B_1 = GRM
A_2 = [0.7 0.0; 0.0 0.7]
B_2 = Matrix{Float64}(I, size(GRM))

#Standard normal

#for multiple glm traits from different distributions
dist_type_vector = [NormalResponse(1), PoissonResponse()]

link_type_vector = [IdentityLink(), LogLink()]


#SINGLE GLM TRAIT
GLM_trait_model_Poisson5 = GLMTrait(formulas[1], df, PoissonResponse(), LogLink())
Simulated_GLM_trait = simulate(GLM_trait_model_Poisson5)


#MULTIPLE GLM TRAITS IID 
Multiple_iid_GLM_traits_model = Multiple_GLMTraits(formulas, df, PoissonResponse(), LogLink())
Simulated_GLM_trait_iid = simulate(Multiple_iid_GLM_traits_model)

#MULTIPLE GLM TRAITS FROM DIFFERENT DISTRIBUTIONS

Multiple_GLM_traits_model_NOTIID = Multiple_GLMTraits(formulas, df, dist_type_vector, link_type_vector)
Simulated_GLM_trait_NOTIID = simulate(Multiple_GLM_traits_model_NOTIID)

#LMM TRAIT
LMM_trait_model = LMMTrait(formulas, df, @vc A_1 ⊗ B_1 + A_2 ⊗ B_2)
Simulated_LMM_trait = simulate(LMM_trait_model)


### TESTING 

# B = 2000
# variancecomp = @vc A_1 ⊗ B_1 + A_2 ⊗ B_2

# function testing(simulatedtrait::Matrix)
# 	n_people = size(simulatedtrait)[1]
# 	n_traits = size(simulatedtrait)[2]

# 	Y = Vector{Matrix{Float64}}(undef, B)
# 	ybar = Vector{Vector{Float64}}(undef, B)
# 	diff = Vector{Matrix{Float64}}(undef, B)
# 	sumabs2 = Vector{Float64}(undef, B)
# for i in 1:B
# 	Y[i] = TraitSimulation.multiple_trait_simulation7(zeros(n_people, n_traits), variancecomp)
# 	ybar[i] = vec(mean(Y[i], dims = 1))
# 	diff[i] = Y[i] - reshape(repeat(ybar[i], inner = n_people), n_people, n_traits)
# 	sumabs2[i] = sum(abs2, sum(diff[i], dims = 1))
# end

# sample_mean = mean(ybar)
# sample_variance = sum(sumabs2)/(B - 1)
# return(sample_mean, sample_variance)
# end

#testing(Matrix(Simulated_GLM_trait))

### TESTING 

B = 2000
variancecomp = @vc A_1 ⊗ B_1 + A_2 ⊗ B_2

function testing_new(n_people, n_traits, variancecomp)
	Y = Vector{Matrix{Float64}}(undef, B)
	ybar = Vector{Vector{Float64}}(undef, B)
	CovMatrices_people = Vector{Matrix{Float64}}(undef, B)
	CovMatrices_traits = Vector{Matrix{Float64}}(undef, B)

for i in 1:B
	Y[i] = TraitSimulation.multiple_trait_simulation7(zeros(n_people, n_traits), variancecomp)
	ybar[i] = vec(mean(Y[i], dims = 1))
	CovMatrices_people[i] = cov(Y[i], dims = 2)
	CovMatrices_traits[i] = cov(Y[i], dims = 1)
end

sample_mean = mean(ybar)
sample_cov_matrix_people = mean(CovMatrices_people)
sample_cov_matrix_traits = mean(CovMatrices_traits)
Total_Sigma = kron(sample_cov_matrix_people, sample_cov_matrix_traits)
return(sample_mean, Total_Sigma)
end

#norm(testing_new(212, 2, variancecomp)[2] - A_1 ⊗ B_1 + A_2 ⊗ B_2)

cd("/Users/sarahji/Desktop/TraitSimulation")