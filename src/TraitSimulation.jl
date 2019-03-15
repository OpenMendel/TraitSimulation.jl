module TraitSimulation
using GLM # this already defines some useful distribution and link types
using DataFrames # so we can test it 
using StatsModels # useful distributions
using Distributions #lots more useful distributions
using LinearAlgebra
using Random

# struct ResponseType{D<:Distributions.Distribution, L<:GLM.Link}
#   family::D
#   inverse_link::L
#   location::Float64
#   scale::Float64
#   shape::Float64
#   df::Float64
#   trials::Int

# # #inner constructor
# #   function ResponseType(family::D, inverse_link::L, location, scale, shape, df, trials) where {D, L}
# #     if !(D isa Distributions.Distribution) 
# #       error("Distribution $(D) is not supported!")
# #     end
# #     if !(L isa GLM.Link)
# #       error("Link $(L) is not supported!")
# #     end
# #     return new{D, L}(family, inverse_link, location, scale, shape, df, trials) #overriding default types in responsetype 
# #   end
# end

# struct NewResponseType{D<:Distributions.Distribution, L<:GLM.Link}
#   family::D
#   inverse_link::L
#   location::Float64
#   scale::Float64
#   shape::Float64
#   df::Float64
#   trials::Int
# end

include("calculate_mean_vector.jl")
#not in glm package say weibull assuming i find out what the weibull link is 
#apply_inverse_link(μ, dist::ResponseType{D, LogLink}) where D = weibull_link.(μ)

#in glm package
#include("apply_inverse_link.jl")
include("apply_inverse_link_new.jl")
export LogLink, IdentityLink, SqrtLink, ProbitLink, LogitLink, InverseLink, CauchitLink, CloglogLink

# #since the GLM package uses the Distribution types from the Distibutions package, we use these packages not to simulate from their existing functions but to 
# #use them as type dispatchers for the simulate_glm_trait function

#include("simulate_glm_trait.jl")

include("simulate_glm_trait_new.jl")
export PoissonResponse, NormalResponse, BinomialResponse, BernoulliResponse, GammaResponse, InverseGaussianResponse, TResponse, WeibullResponse #Exporting these from the Distributions package 


#this is the main functionality of this package, to run the actual simulation.
# function actual_simulation(mu, dist::ResponseType) 
# 	transmu = apply_inverse_link(mu, dist)
# 	Simulated_Trait = simulate_glm_trait(transmu, dist)
# 	return(Simulated_Trait)
# end

#this is the main functionality of this package, to run the actual simulation now for the split up responsedist type and linkfunction type
function actual_simulation(mu, dist::ResponseDistribution, link::InverseLinkFunction) 
  transmu = apply_inverse_link(mu, link)
  Simulated_Trait = simulate_glm_trait(transmu, dist)
  return(Simulated_Trait)
end

########

include("Multiple_traits.jl")

include("Model_Framework.jl")

#this for GLM trait 
function simulate(trait::GLMTrait)
  simulated_trait = actual_simulation(trait.mu, trait.dist, trait.link)
  out = DataFrame(trait1 = simulated_trait)
  return(out)
end

# for multiple GLM traits 
function simulate(traits::Vector)
  simulated_traits = [actual_simulation(traits[i].mu, traits[i].dist, traits[i].link) for i in 1:length(traits)]
  out = DataFrame(simulated_traits)
  out = names!(out, [Symbol("trait$i") for i in 1:length(traits)])
  return(out)
end

# LMMtrait
function simulate(trait::LMMTrait)
  multiple_trait_simulation7(trait.mu, trait.vc)
end

export ResponseType, actual_simulation, mean_formula, VarianceComponent, append_terms!, GLMTrait, Multiple_GLMTraits, LMMTrait, simulate, @vc, g
end #module

