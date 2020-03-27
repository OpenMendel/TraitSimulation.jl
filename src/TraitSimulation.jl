module TraitSimulation
using GLM # this already defines some useful distribution and link types
using DataFrames # so we can test it
using LinearAlgebra
using Random
using SnpArrays
# using SpecialFunctions
using OrdinalMultinomialModels
import Base: show

include("simulatematrixnormal.jl")

include("modelparameterparsers.jl")

include("modelframework.jl")

include("simulatepower.jl")

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
      return dist(1, 1 / (1 + μ)) # r = 1
  end

  # clamp the trait values
    # specific to Poisson and NegativeBinomial
    function clamp_eta(X, β, distT::Union{Type{Poisson}, Type{NegativeBinomial}}; lb, ub)
    	systematic_component = map(y -> y >= ub ? ub : y, X*β)
    	return systematic_component
      end

      # specific to Bernoulli and Binomial
      function clamp_eta(X, β, distT::Union{Type{Bernoulli}, Type{Binomial}}; lb , ub)
    	  systematic_component = X*β
    	  return clamp!(systematic_component, lb, ub)
      end

      function clamp_glm(trait::GLMTrait; lb = -20, ub = 20)
    	  trait.η .= clamp_eta(trait.X, trait.β, distT; lb = lb, ub = ub)
    	  for i in eachindex(trait.μ)
    		  trait.μ[i] = GLM.linkinv(trait.link, trait.η[i])
    	  end
        return trait
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
          @views simulate!(Y[:, k], trait)
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
     Y = zeros(size(trait.mu)) # preallocate
     simulate!(Y, trait) # do the simulation
     return Y
  end

  function simulate!(Y, trait::VCMTrait)
      VCM_trait_simulation(Y, trait.mu, trait.vc)
      return Y
  end

  """
  ```
  simulate(trait::VCMTrait, n::Integer)
  ```
  This simulates (a) trait(s), n times under the desired variance component model, specified using the VCMTrait type.
  """
  function simulate(trait::VCMTrait, n::Integer)
      # pre-allocate output
      Y_n = ntuple(x -> zeros(size(trait.mu)), n)
      # do the simulation n times, storing each result in a column
      for k in 1:n
          @views simulate!(Y_n[k], trait)
      end
      return Y_n
  end

  function simulate(trait::GLMMTrait)
      # pre-allocate output
      Y = Matrix{eltype(trait.dist)}(undef, nsamplesize(trait), ntraits(trait))
      # do the simulation
      simulate!(Y, trait)
      return Y
  end

  function simulate!(Y, trait::GLMMTrait)
      #  simulate random effects
      fill!(trait.Z, 0)
      VCM_trait_simulation(trait.Z, trait.η, trait.vc)
      # simulate from the glm with the mean μ and vector of ones for coefficients
      trait.μ .= linkinv.(trait.link, trait.Z)
      @.  Y = rand(__get_distribution(trait.dist, trait.μ))
      return Y
  end

  """
  ```
  simulate(trait::VCMTrait, n::Integer)
  ```
  This simulates (a) trait(s), n times under the desired variance component model, specified using the VCMTrait type.
  """
  function simulate(trait::GLMMTrait, n::Integer)
      # pre-allocate output
      Y_n = ntuple(x -> zeros(size(trait.μ)), n)
      # do the simulation n times, storing each result
      for k in 1:n
          @views simulate!(Y_n[k], trait)
      end
      return Y_n
  end

  export mean_formula, VarianceComponent, TotalVarianceComponent, GLMMTrait
  export GLMTrait, OrderedMultinomialTrait, VCMTrait, simulate, @vc, vcobjtuple
  export simulate_effect_size, snparray_simulation, genotype_sim
  export power_simulation, power, ordinal_power_simulation, clamp_glm
end #module
