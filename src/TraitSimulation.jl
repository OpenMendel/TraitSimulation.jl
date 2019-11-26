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

include("rand.jl")


  """
  ```
  simulate(trait, n_reps)
  ```
  this for simulating a single GLM trait, n_reps times. 
  """
  function simulate(glmtraitmodel::GLMTrait)
      simulated_trait = rand(GLMTrait.responsedist)
      return(simulated_trait)
  end

  function simulate(glmtraitmodel::GLMTrait, n_reps::Int64)
    n_people = length(glmtraitmodel.mu)
    rep_simulation = Vector{Float64}(undef, n_reps)
    for i in 1:n_reps
      rep_simulation[i] = simulate(glmtraitmodel)
    end
      return(rep_simulation)
  end

  function simulate(glmtraitmodel::Vector{GLMTrait}, n_reps::Int64)
    n_traits = length(glmtraitmodel)
    rep_simulation = Matrix{Float64}(undef, n_reps, n_traits)
    for i in 1:n_traits
      for j in 1:n_reps
        rep_simulation[j, i] = simulate(glmtraitmodel[i])
      end
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
  export GLMTrait, Multiple_GLMTraits, LMMTrait, VCM_simulation, simulate, @vc, vcobjtuple, SimulateMVN, SimulateMVN!, Aggregate_VarianceComponents
  export Generate_Random_Model_Chisq

end #module
