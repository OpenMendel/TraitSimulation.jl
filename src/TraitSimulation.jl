module TraitSimulation
using GLM # this already defines some useful distribution and link types
using DataFrames # so we can test it 
using StatsModels # useful distributions #lots more useful distributions
using LinearAlgebra
using Random
using SpecialFunctions

include("calculate_mean_vector.jl")

include("apply_inverse_link_new.jl")
export LogLink, IdentityLink, SqrtLink, ProbitLink, LogitLink, InverseLink, CauchitLink, CloglogLink

include("simulate_glm_trait_new.jl")
export PoissonResponse, NormalResponse, BinomialResponse, BernoulliResponse, GammaResponse, InverseGaussianResponse, TResponse, WeibullResponse #Exporting these from the Distributions package 


#this is the main functionality of this package, to run the actual simulation now for the split up responsedist type and linkfunction type
function GLM_trait_simulation(mu, dist::ResponseDistribution, link::InverseLinkFunction) 
  transmu = apply_inverse_link(mu, link)
  Simulated_Trait = simulate_glm_trait(transmu, dist)
  return(Simulated_Trait)
end

########

include("Multiple_traits.jl")

include("Model_Framework.jl")

"""
```
simulate(trait, n_reps)
```
this for simulating a single GLM trait, n_reps times. 
"""
function simulate(trait::GLMTrait)
    simulated_trait = GLM_trait_simulation(trait.mu, trait.dist, trait.link)
    #rep_simulation = DataFrame(trait1 = simulated_trait)
    return(simulated_trait)
end

function simulate(trait::GLMTrait, n_reps::Int64)
  n_people = length(trait.mu)
  rep_simulation = Matrix{Float64}(undef, n_people, n_reps)
  for i in 1:n_reps
    rep_simulation[:, i] .= simulate(trait) # store each data frame in the vector of dataframes rep_simulation
  end
    return(rep_simulation)
end

"""
```
simulate(trait, nreps)
```
this for simulating multiple LMMtraits, n_reps times. 
"""
function simulate(trait::LMMTrait)
  rep_simulation = LMM_trait_simulation(trait.mu, trait.vc)
  return(rep_simulation)
end

function simulate(trait::LMMTrait, n_reps::Int64)
  n_people, n_traits = size(trait.mu)
  rep_simulation = zeros(n_people, n_traits, n_reps)
  for i in 1:n_reps
    rep_simulation[:, :, i] = simulate(trait)
  end
  return(rep_simulation)
end


export ResponseType, GLM_trait_simulation, mean_formula, VarianceComponent, append_terms!, LMM_trait_simulation
export GLMTrait, Multiple_GLMTraits, LMMTrait, simulate, @vc, vcobjtuple, SimulateMVN, SimulateMVN!, Aggregate_VarianceComponents!
export TResponse, WeibullResponse, PoissonResponse, NormalResponse, BernoulliResponse, BinomialResponse
export GammaResponse, InverseGaussianResponse, ExponentialResponse
export CauchitLink, CloglogLink, IdentityLink, InverseLink, LogitLink, LogLink, ProbitLink, SqrtLink
end #module

