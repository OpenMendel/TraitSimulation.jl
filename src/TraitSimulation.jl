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

function actual_simulation(mu, dist::ResponseType)
	transmu = apply_inverse_link(mu, dist)
	Simulated_Trait = simulate_glm_trait(transmu, dist)
	return(Simulated_Trait)
end

export ResponseType, actual_simulation, mean_formula
end # module
