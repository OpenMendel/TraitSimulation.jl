using DataFrames, Distributions, SnpArrays, TraitSimulation, StatsModels, Random, LinearAlgebra

#cd /Users/sarahji/Desktop/OpenMendel_Sarah/Tutorials/Heritability 

# 1. DATA IMPORT:
#reads the bed file of snp data and then turns into named dataframe for simulation
cd /Users/sarahji/Desktop/OpenMendel_Sarah/Tutorials/Heritability 
snps = SnpArray("heritability.bed")
minor_allele_frequency = maf(snps)
common_snps_index = (0.05 .≤ minor_allele_frequency)
common_snps = SnpArrays.filter("heritability", trues(212), common_snps_index)
df = convert(Matrix{Float64}, snps)
df = DataFrame(df)


# glm: for one trait at a time (Exponential Family)
# specifying multiple formulas can be done for a vector of GLMTrait objects 
struct GLMTrait
formula::String
mu::Vector{Float64}
dist:: ResponseType
function GLMTrait(formula, df, dist)
		mu = mean_formula(formula, df)
		return(new(formula, mu, dist))
	end
end

function Multiple_GLMTraits(formulas::Vector{String}, df::DataFrame, dist::ResponseType)
	vec = [GLMTrait(formulas[i], df, dist) for i in 1:length(formulas)]
	return(vec)
end


struct ResponseType{D<:Distributions.Distribution, L<:GLM.Link}
  family::D
  inverse_link::L
  location::Float64
  scale::Float64
  shape::Float64
  df::Float64
  trials::Int
  function ResponseType(dist)
		return(new(dist))
	end
end

#NEED to create a function that returns a vector of ResponseTypes to feed into the Multiple_GLM_traits_model_NOTIID
function Multiple_ResponseTypes(formulas::Vector{String}, df::DataFrame, dist::ResponseType)
	vec = [GLMTrait(formulas[i], df, dist) for i in 1:length(formulas)]
	return(vec)
end


#so that user can simulate simultaneous glm traits from different distributions from exponential family
#this function must return a vector of GLM traits simulated from a vector of ResponseType's  
function Multiple_GLMTraits(formulas::Vector{String}, df::DataFrame, dist::Vector{ResponseType})
	vec = [GLMTrait(formulas[i], df, dist[i]) for i in 1:length(formulas)]
	return(vec)
end


#this for GLM trait 
function simulate(trait::GLMTrait)
	simulated_trait = actual_simulation(trait.mu, trait.dist::ResponseType)
	out = DataFrame(trait1 = simulated_trait)
	return(out)
end

# for GLM traits multiple 
function simulate(traits::Vector{GLMTrait})
	simulated_traits = [actual_simulation(traits[i].mu, traits[i].dist) for i in 1:length(traits)]
	out = DataFrame(simulated_traits)
	out = names!(out, [Symbol("trait$i") for i in 1:length(traits)])
	return(out)
end

# 2. MODEL SPECIFICATION:
# Mean specification for mu: 
formulas = ["1 + 3(x1)", "1 + 3(x2) + abs(x3)"]

# Variance Specification for VCM: ex) @vc A ⊗ GRM + B ⊗ I
GRM = grm(common_snps)
A_1 = [0.3 0.1; 0.1 0.3]
B_1 = GRM
A_2 = [0.7 0.0; 0.0 0.7]
B_2 = Matrix{Float64}(I, size(GRM))


test2 = @vc A_2 ⊗ Matrix{Float64}(I, 212)

#Standard normal
dist_N01 = ResponseType(Normal(), IdentityLink(), 0.0, 1.0, 0.0, 0.0, 0)

#Normal(0, 5)
dist_N05 = ResponseType(Normal(), IdentityLink(), 0.0, 5.0, 0.0, 0.0, 0)

#for multiple glm traits from different distributions
dist_type_vector = [dist_N01, dist_N05]

#LITTLE TEST:
# this works fine, I need to figure out what's happening with Multiple_GLMTraits()

#creates a vector of GLM trait objects
#vec_test = [GLMTrait(formulas[1], df, dist_type_vector[1]), GLMTrait(formulas[2], df, dist_type_vector[2])]
#simulate(vec_test)

#SINGLE GLM TRAIT
GLM_trait_model = GLMTrait(formulas[1], df, dist_N01)
Simulated_GLM_trait = simulate(GLM_trait_model)


#MULTIPLE GLM TRAITS IID 
Multiple_iid_GLM_traits_model = Multiple_GLMTraits(formulas, df, dist_N01)
Simulated_GLM_trait = simulate(Multiple_iid_GLM_traits_model)

#MULTIPLE GLM TRAITS FROM DIFFERENT DISTRIBUTIONS

Multiple_GLM_traits_model_NOTIID = Multiple_GLMTraits(formulas, df, dist_type_vector)
Simulated_GLM_trait = simulate(Multiple_GLM_traits_model_NOTIID)

#trait1, trait2 = (simulate(GLMTrait(formulaszzz[1], df, dist)), simulate(GLMTrait(formulaszzz[2], df, dist)))
# lmm: multiple traits (MVN)

struct LMMTrait
formulas::Vector{String}
mu::Matrix{Float64}
vc::Vector{VarianceComponent}
	function LMMTrait(formulas, df, vc)
		n_traits = length(formulas)
		n_people = size(df)[1]
		mu = zeros(n_people, n_traits)
		for i in 1:n_traits
			#calculate the mean vector
			mu[:, i] += mean_formula(formulas[i], df)
		end
		return(new(formulas, mu, vc))
	end
end

#idea 
function simulate(trait::LMMTrait)
	multiple_trait_simulation7(trait.mu, trait.vc)
end


#LMM TRAIT
LMM_trait_model = LMMTrait(formulas, df, @vc A_1 ⊗ B_1 + A_2 ⊗ B_2)
Simulated_LMM_trait = simulate(LMM_trait_model)

#### TESTING 
B = 2000
variancecomp = @vc A_1 ⊗ B_1 + A_2 ⊗ B_2


function testing(trait::LMMTrait)
	Y = Vector{Matrix{Float64}}(undef, 2000)
	ybar = zeros(B, length(trait.formulas))
	diff = zeros()
for i in 1:B
	for j in 1:length(trait.formulas)
	Y[i] = multiple_trait_simulation7(zeros(212, 2), variancecomp)
	ybar[i, j] = mean(Y[i][:, j])
	diff = (Y[i][:, j] - ybar[i, j])
end
sample_variance = sum(abs2, diff)
end

ybar = mean(Y)
sample_variance = sum(abs2, Y - ybar)


