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

include("orderedmultinomialpower.jl")

include("simulatesnparray.jl")


  function simulate(trait::GLMTrait)
      # pre-allocate output
      y = Vector{eltype(trait.dist)}(undef, nsamplesize(trait))
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

  """
  ```
  simulate(trait, n_reps)
  ```
  this for simulating a single univariate trait, n_reps times. By default the simulate(trait) function will run a single simulation
  """
  function simulate(trait::GLMTrait, n::Integer)
      # pre-allocate output
      Y = Matrix{eltype(trait.dist)}(undef, nsamplesize(trait), n)

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
     y = Vector{Int64}(undef, nsamplesize(trait)) # preallocate
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
      Y = Matrix{Int64}(undef, nsamplesize(trait), n)
      # do the simulation n times, storing each result in a column
      for k in 1:n
          @views simulate!(Y[:, k], trait; Logistic = Logistic, threshold = threshold)
      end
      return Y
  end

  function simulate(trait::VCMTrait)
     Y = Matrix{eltype(trait.mu)}(undef, size(trait.mu)) # preallocate
     simulate!(Y, trait) # do the simulation
     return Y
  end

  function simulate!(Y, trait::VCMTrait)
      Y .= VCM_trait_simulation(trait.mu, trait.vc)
      return Y
  end

  function simulate(trait::VCMTrait, n::Integer)
      # pre-allocate output
      Y_n = ntuple(x -> zeros(size(trait.mu)), n)
      # do the simulation n times, storing each result in a column
      for k in 1:n
          @views simulate!(Y_n[k], trait)
      end
      return Y_n
  end


  export mean_formula, VarianceComponent, VCM_trait_simulation
  export GLMTrait, OrderedMultinomialTrait, VCMTrait, simulate, @vc, vcobjtuple
  export simulate_effect_size, snparray_simulation, genotype_sim, realistic_multinomial_powers, power_multinomial_models
  export ordinal_multinomial_power, power, realistic_power_simulation
end #module
