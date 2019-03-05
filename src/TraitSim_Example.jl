using DataFrames, Distributions, SnpArrays, TraitSimulation, StatsModels

#cd /Users/sarahji/Desktop/OpenMendel_Sarah/Tutorials/Heritability 

# 1. DATA IMPORT:
#reads the bed file of snp data and then turns into named dataframe for simulation
snps = SnpArray("heritability.bed")
minor_allele_frequency = maf(snps)
common_snps_index = (0.05 .≤ minor_allele_frequency)
common_snps = SnpArrays.filter("heritability", trues(212), common_snps_index)
df = convert(Matrix{Float64}, snps)
df = DataFrame(df)


# 2. MODEL SPECIFICATION:
# Mean specification for mu: 
formulaszzz = ["1 + 3(x1)", "1 + 3(x2) + abs(x3)"]

# Variance Specification for VCM: ex) @vc A ⊗ GRM + B ⊗ I
GRM = grm(common_snps)
A_1 = [0.3 0.1; 0.1 0.3]
B_1 = GRM
A_2 = [0.7 0.0; 0.0 0.7]
B_2 = Matrix{Float64}(I, size(GRM))

sim_model = (μ, ,IdentityLink(), NormalResponse(1.0))


struct VarianceComponent
  # Stores a single variance component, a vector of variance component
  #for a number of traits, or a cross covariance matrix
  var_comp::Matrix{Float64}
  # stores the covariance matrix
  cov_mat::Matrix{Float64}
end 


# glm: one trait (Exponential Family)

struct GLMTrait <: SimulationModel
mean_formula
response_dist:: ResponseType
end

function GLMTrait(formulas, df, ResponseType)
end


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






multiple_trait_simulation5(formulas, dataframe, A, B)
