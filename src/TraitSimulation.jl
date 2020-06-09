module TraitSimulation
using GLM # this already defines some useful distribution and link types
using DataFrames # so we can test it
using LinearAlgebra
using Random
using SnpArrays
using OrdinalMultinomialModels
using VarianceComponentModels
using Distributions
import Base: show

include("simulatematrixnormal.jl")

include("modelframework.jl")

include("simulatesnparray.jl")

include("modelparameterparsers.jl")

include("simulatepower.jl")

"Simulate the trait under the given model."
simulate(trait::AbstractTraitModel) = __default_behavior(trait)

"Simulate a trait `n` times independently."
simulate(trait::AbstractTraitModel, n::Integer) = __default_behavior(trait)

"Simulate a trait and store the result in y."
simulate!(y, trait::AbstractTraitModel) = __default_behavior(trait)

  function simulate(trait::GLMTrait)
      # pre-allocate output
      y = Vector{eltype(trait.dist)}(undef, nsamplesize(trait))
      # do the simulation
      simulate!(y, trait)
      return y
  end

  function simulate!(y, trait::GLMTrait)
      dist = trait.dist
      for i in eachindex(y)
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
      β = 1 / μ # here β is the rate parameter for gamma distribution
      return dist(1, β) # α = 1
  end

  # specific to NegativeBinomial
  function __get_distribution(dist::Type{NegativeBinomial}, μ)
      p = inv(1 + μ)
      return dist(1, p) # r = 1
  end

  """
  ```
  simulate(trait::GLMTrait, n::Integer)
  ```
  This simulates a GLM trait n times under the desired generalized linear model, specified using the GLMTrait type.
  """
  function simulate(trait::GLMTrait, n::Integer)
      # pre-allocate output
      Y = Matrix{eltype(trait.dist)}(undef, nsamplesize(trait), n)
      # do the simulation n times, storing each result in a column
      for k in 1:n
          simulate!(Y[:, k], trait)
      end
      return Y
  end

  """
  ```
  simulate(OrderedMultinomialModel, n_reps; Logistic = false, threshold == empty)
  ```
    This simulates a OrderedMultinomialTrait trait n times under the desired model, specified using the OrderedMultinomialTrait type.
  This simulates a Ordered Multinomial trait n times, under the specifed ordinal multinomial linear model specified as a GLMTrait object for simulation.
  By default we simulate the multinomial ordered outcome, but with the specification of the Logistic and threshold arguments, we can do the transformation to ordinal logistic.
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
          y .= Int64.(y .> threshold) # need threshold/ cutoff for case/control
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
     Y = zeros(size(trait.μ)) # preallocate
     simulate!(Y, trait) # do the simulation
     return Y
  end

  function simulate!(Y, trait::VCMTrait)
      fill!(Y, 0.0)
      TraitSimulation.VCM_trait_simulation(Y, trait.Z, trait.μ, trait.vc)
      return Y
  end

  function simulate_old(trait::VCMTrait, n::Integer)
      # pre-allocate output
      Y_n = [zeros(size(trait.μ)) for _ in 1:n] # replicates
      # do the simulation n times, storing each result in a column
      for k in 1:n
          simulate!(Y_n[k], trait)
      end
      return Y_n
  end

  function simulate(trait::VCMTrait, n::Integer)
      # pre-allocate output
      Y_n = [zeros(size(traitobject.μ)) for _ in 1:n] # holds n replicates
      # do the simulation n times, storing each result in a column
     Threads.@threads for k in 1:n
          simulate!(Y_n[k], trait)
      end
      return simulated_traits
  end

  function simulate(trait::GLMMTrait)
      # pre-allocate output
      Y = Matrix{eltype(trait.dist)}(undef, size(trait.μ))
      # do the simulation
      simulate!(Y, trait)
      return Y
  end

  function simulate!(Y, trait::GLMMTrait)
      #  simulate random effects
      fill!(trait.Z, 0.0)
      fill!(trait.Y_vcm, 0.0)
      TraitSimulation.VCM_trait_simulation(trait.Y_vcm, trait.Z, trait.η, trait.vc)
      # simulate from the glm with the mean μ and vector of ones for coefficients
      trait.μ .= GLM.linkinv.(trait.link, trait.Y_vcm)
      copyto!(Y, rand.(__get_distribution.(trait.dist, trait.μ)))
      Y
  end

  """
  ```
  simulate(trait::GLMMTrait, n::Integer)
  ```
  This simulates a trait n times under the desired generalized linear mixed model, specified using the GLMMTrait type.
  """
  function simulate(trait::GLMMTrait, n::Integer)
      # pre-allocate output
      Y_n = [zeros(size(traitobject.μ)) for _ in 1:n]
      # do the simulation n times, storing each result
      @inbounds for k in 1:n
          @views simulate!(Y_n[k], trait)
      end
      return Y_n
  end

  export mean_formula, VarianceComponent, @vc, vcobjtuple
  export GLMTrait, OrderedMultinomialTrait, VCMTrait, GLMMTrait
  export simulate_effect_size, snparray_simulation, genotype_sim
  export nsamplesize, neffects, noutcomecategories, nvc, ntraits
  export simulate!, simulate
  export null_and_alternative_vcm_and_rotate, power_simulation, power, VCM_trait_simulation
end #module
