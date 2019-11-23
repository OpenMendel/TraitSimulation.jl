module TraitSimulation
using GLM # this already defines some useful distribution and link types
using DataFrames # so we can test it 
using StatsModels # useful distributions #lots more useful distributions
using LinearAlgebra
using Random
using SpecialFunctions
using SnpArrays
using Revise

include("calculate_mean_vector.jl")

include("Multiple_traits.jl")

include("Model_Framework.jl")

include("Simulate_RandomGLM.jl")

#With GLMTrait input type 
"""
GLM_trait_simulation()
Simulates univariate GLM trait from a GLMTrait object.
First transforms the XB mean of fixed effects using the inverse link,
and then passes the transformed mean μ and it's link function into the actual simulation function, glm_simulation.
"""
function GLM_trait_simulation(GLMTraitobj::GLMTrait)
  μ = GLM.linkinv.(GLMTraitobj.link, GLMTraitobj.mu) # get the transformed mean
  Simulated_Trait = GLM_trait_simulation(μ, GLMTraitobj.dist)
  return(Simulated_Trait)
end


function GLM_trait_simulation(μ, dist::Type{NegativeBinomial})
  r = 1
  μ = 1 ./ (1 .+ μ ./ r)
  Simulated_Trait = [rand(dist(r, i)) for i in μ] #number of failtures before r success occurs
  return(Simulated_Trait)
end

function GLM_trait_simulation(μ, dist::Type{Gamma})
  β = 1 ./ μ # here β is the rate parameter for gamma distribution
  α = 1
  Simulated_Trait = [rand(dist(α, i)) for i in β] # α is the shape parameter for gamma
  return(Simulated_Trait)
end

"""
glm_simulation()
Runs the actual simulation of a univariate GLM trait from the transformed mean μ and its response distribution.
"""
function GLM_trait_simulation(μ, dist::D) where D
  Simulated_Trait = [rand(dist(i)) for i in μ]
  return(Simulated_Trait)
end

  ########

  """
  ```
  simulate(trait, n_reps)
  ```
  this for simulating a single GLM trait, n_reps times. 
  """
  function simulate(glmtraitmodel::GLMTrait)
      simulated_trait = GLM_trait_simulation(glmtraitmodel)
      return(simulated_trait)
  end

  function simulate(glmtraitmodel::GLMTrait, n_reps::Int64)
    n_people = length(glmtraitmodel.mu)
    rep_simulation = Matrix{Float64}(undef, n_people, n_reps)
    for i in 1:n_reps
      rep_simulation[:, i] .= simulate(glmtraitmodel) # store each data frame in the vector of dataframes rep_simulation
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
  export GLMTrait, Multiple_GLMTraits, LMMTrait, simulate, @vc, vcobjtuple, SimulateMVN, SimulateMVN!, Aggregate_VarianceComponents
  export Generate_Random_Model_Chisq

end #module
