module TraitSimulation
using GLM # this already defines some useful distribution and link types
using DataFrames # so we can test it
using StatsModels # useful distributions #lots more useful distributions
using LinearAlgebra
using Random
using SpecialFunctions
using OrdinalMultinomialModels

include("calculate_mean_vector.jl")

include("Multiple_traits.jl")

include("Model_Framework.jl")

include("Random_VCM.jl")

include("MultinomialPowerDemo.jl")

include("SnpArraySimulation.jl")

  """
  ```
  simulate(trait, n_reps)
  ```
  this for simulating a single univariate trait, n_reps times.
  """
  function simulate(trait::UnivariateModel)
      # pre-allocate output
      y = Vector{eltype(trait.dist)}(undef, nsamples(trait))
      # do the simulation
      simulate!(y, trait)

      return y
  end

  function simulate!(y, trait::UnivariateModel)
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

  function simulate(trait::UnivariateModel, n::Integer)
      # pre-allocate output
      Y = Matrix{eltype(trait.dist)}(undef, nsamples(trait), n)

      # do the simulation n times, storing each result in a column
      for k in 1:n
          @views simulate!(Y[:, k], trait)
      end

      return Y
  end
  
  # function simulate(trait::OrderedMultinomialModel; Logistic::Bool = false, threshold::Union{T, Nothing} = nothing) where T <: Real
  #     Y = rpolr(trait.X, trait.β, trait.θ, trait.link)
  #     if Logistic
  #       threshold == nothing && error("I need the cutoff for case/control")
  #       Y .= Int64.(Y .> threshold) #makes J/2 the default cutoff for case/control
  #     end
  #     return Y
  # end

  # function simulate(trait::OrderedMultinomialModel)
  #     Y = rpolr(trait.X, trait.β, trait.θ, trait.link)
  #     if Logistic
  #       threshold == nothing && error("I need the cutoff for case/control")
  #       Y .= Int64.(Y .> threshold) #makes J/2 the default cutoff for case/control
  #     end
  #     return Y
  # end

  """
  ```
  simulate(OrderedMultinomialModel, n_reps)
  ```
  this for simulating a single Ordinal trait, n times.
  """
  function simulate(trait::OrderedMultinomialModel)
      # do the simulation
     y = Vector{Int64}(undef, nsamples(trait))
     simulate!(y, trait)
     return y
  end

  function simulate!(y, trait::OrderedMultinomialModel)
      # in a for-loop
      y = rpolr(trait.X, trait.β, trait.θ, trait.link)
      return y
  end

  function simulate(trait::OrderedMultinomialModel, n::Integer)
      # pre-allocate output
      Y = Matrix{Int64}(undef, nsamples(trait), n)

      # do the simulation n times, storing each result in a column
      for k in 1:n
          @views simulate!(Y[:, k], trait)
      end

      return Y
  end


  """
  ```
  simulate(trait, nreps)
  ```
  this for simulating multiple LMMtraits, n_reps times.
  """
  function simulate(trait::VarianceComponentModel)
    rep_simulation = LMM_trait_simulation(trait.mu, trait.vc)
    return(rep_simulation)
  end

  function simulate(trait::VarianceComponentModel, n_reps::Int64)
    n_people, n_traits = size(trait.mu)
    rep_simulation = zeros(n_people, n_traits, n_reps)
    for i in 1:n_reps
      rep_simulation[:, :, i] = simulate(trait)
    end
    return(rep_simulation)
  end

  export ResponseType, GLM_trait_simulation, mean_formula, VarianceComponent, LMM_trait_simulation
  export UnivariateModel, OrderedMultinomialModel, VarianceComponentModel, simulate, @vc, vcobjtuple
  export generateRandomVCM, CompareWithJulia
  export simulate_effect_size, snparray_simulation, genotype_sim, realistic_multinomial_powers, power_multinomial_models
  export realistic_multinomial_power, power, realistic_power_simulation

end #module
