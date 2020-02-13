module TraitSimulation
using GLM # this already defines some useful distribution and link types
using DataFrames # so we can test it
using StatsModels # useful distributions #lots more useful distributions
using LinearAlgebra
using Random
using SpecialFunctions
using OrdinalMultinomialModels

include("meanformulaparser.jl")

include("variancecomponents.jl")

include("modelframework.jl")

include("randvcm.jl")

include("orderedmultinomialpower.jl")

include("simulatesnparray.jl")

  """
  ```
  simulate(trait, n_reps)
  ```
  this for simulating a single univariate trait, n_reps times.
  """
  function simulate(trait::GLMTrait)
      # pre-allocate output
      y = Vector{eltype(trait.dist)}(undef, nsamples(trait))
      # do the simulation
      simulate!(y, trait)

      return y
  end

  function simulate!(y, trait::GLMTrait)
      dist = trait.dist

      # in a for-loop
      for i in eachindex(y)
          # push the work of forming a distribution to a
          # helper function yet-to-be defined
          y[i] = rand(__get_distribution(dist, trait.μ[i]))
      end
      return y
  end

  # default behavior for UnivariateDistribution
  function __get_distribution(dist::Type{D}, μ) where D <: UnivariateDistribution
      return dist(μ)
  end

  # specific to Gamma
  function __get_distribution(dist::Type{Gamma}, μ)
      return dist(1, 1 / (1 + μ)) # r = 1
  end

  # specific to NegativeBinomial
  function __get_distribution(dist::Type{NegativeBinomial}, μ)
      return dist(1, 1 / (1 + μ)) # r = 1
  end

  function simulate(trait::GLMTrait, n::Integer)
      # pre-allocate output
      Y = Matrix{eltype(trait.dist)}(undef, nsamples(trait), n)

      # do the simulation n times, storing each result in a column
      for k in 1:n
          @views simulate!(Y[:, k], trait)
      end

      return Y
  end

  """
  ```
  simulate(OrderedMultinomialModel, n_reps; Logistic = false, threshold == empty)
  ```
  this for simulating a single Ordinal trait, n times. By default we simulate the multinomial ordered outcome, but with the specification of the Logistic and threshold arguments, we can do the transformation to ordinal logistic.
  """
  function simulate(trait::OrderedMultinomialTrait; Logistic::Bool = false, threshold::Union{T, Nothing} = nothing) where T <: Real
     y = Vector{Int64}(undef, nsamples(trait)) # preallocate
     simulate!(y, trait; Logistic = Logistic, threshold = threshold) # do the simulation
     return y
  end

  function simulate!(y, trait::OrderedMultinomialTrait; Logistic::Bool = false, threshold::Union{T, Nothing} = nothing) where T <: Real
      # in a for-loop
      y .= rpolr(trait.X, trait.β, trait.θ, trait.link)
      if Logistic
          threshold == nothing && error("I need the cutoff for case/control")
          y .= Int64.(y .> threshold) #makes J/2 the default cutoff for case/control
      end
      return y
  end

  function simulate(trait::OrderedMultinomialTrait, n::Integer; Logistic::Bool = false, threshold::Union{T, Nothing} = nothing) where T <: Real
      # pre-allocate output
      Y = Matrix{Int64}(undef, nsamples(trait), n)
      # do the simulation n times, storing each result in a column
      for k in 1:n
          @views simulate!(Y[:, k], trait; Logistic = Logistic, threshold = threshold)
      end

      return Y
  end


  """
  ```
  simulate(trait, nreps)
  ```
  this for simulating multiple VarianceComponentModel, n_reps times.
  """
  function simulate(trait::VCMTrait)
    rep_simulation = LMM_trait_simulation(trait.mu, trait.vc)
    return(rep_simulation)
  end

  function simulate(trait::VCMTrait, n_reps::Int64)
    n_people, n_traits = size(trait.mu)
    rep_simulation = zeros(n_people, n_traits, n_reps)
    for i in 1:n_reps
      rep_simulation[:, :, i] = simulate(trait)
    end
    return(rep_simulation)
  end

  export mean_formula, VarianceComponent, LMM_trait_simulation
  export GLMTrait, OrderedMultinomialTrait, VCMTrait, simulate, @vc, vcobjtuple
  export generateRandomVCM, CompareWithJulia
  export simulate_effect_size, snparray_simulation, genotype_sim, realistic_multinomial_powers, power_multinomial_models
  export realistic_multinomial_power, power, realistic_power_simulation

end #module
