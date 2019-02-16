module TraitSimulation
using GLM # this already defines some useful distribution and link types
using DataFrames # so we can test it 
using StatsModels # useful distributions
using Distributions #lots more useful distributions

struct ResponseType{D<:Distributions.Distribution, L<:GLM.Link}
  family::D
  inverse_link::L
  location::Float64
  scale::Float64
  shape::Float64
  df::Float64
  trials::Int

# #inner constructor
#   function ResponseType(family::D, inverse_link::L, location, scale, shape, df, trials) where {D, L}
#     if !(D isa Distributions.Distribution) 
#       error("Distribution $(D) is not supported!")
#     end
#     if !(L isa GLM.Link)
#       error("Link $(L) is not supported!")
#     end
#     return new{D, L}(family, inverse_link, location, scale, shape, df, trials) #overriding default types in responsetype 
#   end

end


include("calculate_mean_vector.jl")
#not in glm package say weibull assuming i find out what the weibull link is 
#apply_inverse_link(μ, dist::ResponseType{D, LogLink}) where D = weibull_link.(μ)

#in glm package
include("apply_inverse_link.jl")
export LogLink, IdentityLink, SqrtLink, ProbitLink, LogitLink, InverseLink, CauchitLink, CloglogLink

# #since the GLM package uses the Distribution types from the Distibutions package, we use these packages not to simulate from their existing functions but to 
# #use them as type dispatchers for the simulate_glm_trait function

include("simulate_glm_trait.jl")
export Poisson, Normal, Binomial, Bernoulli, Gamma, InverseGaussian, TDist, Weibull #Exporting these from the Distributions package 


#this is the main functionality of this package, to run the actual simulation.
function actual_simulation(mu, dist::ResponseType) 
	transmu = apply_inverse_link(mu, dist)
	Simulated_Trait = simulate_glm_trait(transmu, dist)
	return(Simulated_Trait)
end

export ResponseType, actual_simulation, mean_formula
end # module

#Toy example test
df = DataFrame(x1 = rand(10000), x2 = rand(10000), x3 = rand(10000), x4 = rand(10000))
user_formula_string = "3 + log(x1) + 2sqrt(abs(log(x2))) + asin(x4)"


μ = mean_formula(user_formula_string, df)

# # # #Poisson
dist = ResponseType(Poisson(), LogLink(), 0.0, 0.0, 0.0, 0.0, 0)
# dist = ResponseType(Poisson(), 2, 2.0, 2.0, 0.0, 0.0, 0)
# actual_simulation(μ, dist)

#kens yellow book chapter 8
#correlation between snps in snparrays using snparrays and store into a sparse matrix
