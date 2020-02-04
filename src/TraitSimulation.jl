module TraitSimulation
using GLM # this already defines some useful distribution and link types
using DataFrames # so we can test it
using StatsModels # useful distributions #lots more useful distributions
using LinearAlgebra
using Random
using SpecialFunctions

include("calculate_mean_vector.jl")

include("Multiple_traits.jl")

include("Model_Framework.jl")

include("RVCModel.jl")

include("SnpArraySimulation.jl")

  """
  ```
  simulate(trait, n_reps)
  ```
  this for simulating a single GLM trait, n_reps times.
  """
  function simulate(trait::GLMTrait)
      simulated_trait = rand.(trait.responsedist)
      return(simulated_trait)
  end

  function simulate(trait::GLMTrait, n_reps::Int64)
    n_people = length(trait.mu)
    T = eltype(trait.responsedist)
    rep_simulation = Vector{T}(undef, n_reps)
    for i in 1:n_reps
      rep_simulation[i] = simulate(trait)
    end
      return(rep_simulation)
  end


  """
  ```
  simulate(OrdinalTrait)
  ```
  this for simulating a single Ordinal trait once.
  """
  function simulate(trait::OrdinalTrait; Logistic::Bool = false, threshold::Union{T, Nothing} = nothing) where T <: Real
      simulated_trait = rpolr(trait.X, trait.β, trait.θ, trait.link)
      if Logistic
        threshold == nothing && error("I need the cutoff for case/control")
        simulated_trait .= Int64.(simulated_trait .> threshold) #makes J/2 the default cutoff for case/control
      end
      return(simulated_trait)
  end
# simulate.(uvec, Logistic = true, threshold = threshold)
  """
  ```
  simulate(OrdinalTrait, n_reps)
  ```
  this for simulating a single Ordinal trait, n_reps times.
  """

  function simulate(trait::OrdinalTrait, n_reps::Int64)
    n_people = size(trait.X, 1)
    rep_simulation = Vector{T}(undef, n_reps)
    for i in 1:n_reps
      rep_simulation[i] = simulate(trait)
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
  export GLMTrait, OrdinalTrait, LMMTrait, VCM_simulation, simulate, @vc, vcobjtuple, SimulateMVN, SimulateMVN!, Aggregate_VarianceComponents
  export Generate_Random_Model_Chisq, RVCModel, CompareWithJulia
  export simulate_effect_size, snparray_simulation, genotype_sim, realistic_multinomial_powers

end #module
