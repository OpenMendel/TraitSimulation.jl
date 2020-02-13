using GLM, Statistics
import GLM: linkinv, linkfun
import TraitSimulation: @vc, VarianceComponent
import Distributions: UnivariateDistribution

# root type: use this to define an interface shared by
# all instances of AbstractTrait
abstract type AbstractTraitModel end

# for example, you might want to be able to check
# the number of samples or linear predictors

__default_behavior(trait) = error("function not supported by $(typeof(trait))")

"Check the number of observations."
nsamples(trait::AbstractTraitModel) = __default_behavior(trait)

"Check the number of linear predictors."
neffects(trait::AbstractTraitModel) = __default_behavior(trait)

"Simulate the trait under the given model for each observation independently."
simulate(trait::AbstractTraitModel) = __default_behavior(trait)

"Simulate a trait `n` times (independently)."
simulate(trait::AbstractTraitModel, n::Integer) = __default_behavior(trait)

"Simulate a trait and store the result in `y`."
simulate!(y, trait::AbstractTraitModel) = __default_behavior(trait)


# Now let's define our first concrete type.

struct UnivariateModel{distT,linkT,vecT1,vecT2,matT} <: AbstractTraitModel
    X::matT             # all effects
    β::vecT1            # regression coefficients
    η::vecT2            # linear predictor η = X*β
    μ::vecT2            # expected value of response μ = g^-1(η)  where g is the link function
    dist::Type{distT}   # univariate, exponential family of distributions
    link::linkT         # link function g(μ) = X*β = η

    function UnivariateModel(X::matT, β::vecT1, η::vecT2, μ::vecT2, distribution::D, link::linkT) where {D,linkT,vecT1,vecT2,matT}
        # extract the base type without type parameters
        distT = Base.typename(typeof(distribution)).wrapper

        # make a new instance
        new{distT,linkT,vecT1,vecT2,matT}(X, β, η, μ, distT, link)
    end
end

# define outer constructors that act as intermediates between the internal
# constructor and any external interfaces we deem convenient

# building from model encoded as mat-vec
function UnivariateModel(X::AbstractMatrix, β::AbstractVector, distribution, link)
    # define the linear predictor
    η = X * β

    # apply the inverse link element-wise
    μ = linkinv.(link, η)

    # create a new instance
    UnivariateModel(X, β, η, μ, distribution, link)
end

# building from linear predictor only
function UnivariateModel(x::AbstractVector, distribution, link, ismu::Bool = true)
    if ismu
        μ = x
        η = linkfun.(link, μ)
    else
        η = x
        μ = linkinv.(link, η)
    end

    return UnivariateModel(nothing, nothing, η, μ, distribution, link)
end

# better printing; customize how a type is summarized in a REPL
import Base: show

function show(io::IO, x::UnivariateModel)
    print(io, "Univariate Model\n")
    print(io, "  * response distribution: $(x.dist)\n")
    print(io, "  * link function: $(typeof(x.link))\n")
    print(io, "  * sample size: $(nsamples(x))")
end

# make our new type implement the interface defined above
nsamples(trait::UnivariateModel) = length(trait.μ)
neffects(trait::UnivariateModel) = size(trait.X, 2)


struct OrderedMultinomialModel{matT,vecT1,vecT2, linkT} <: AbstractTraitModel
    X::matT             # all effects
    β::vecT1            # regression coefficients
    θ::vecT2            # must be increasing
	link::linkT
	function OrderedMultinomialModel(X::matT, β::vecT1, θ::vecT2, link::linkT)  where {matT, vecT1, vecT2, linkT}
    return new{matT, vecT1, vecT2, linkT}(X, β, θ, link)
  end
end

function show(io::IO, x::OrderedMultinomialModel)
    print(io, "Ordinal Multinomial Model\n")
    print(io, "  * number of fixed effects: $(neffects(x))\n")
	print(io, "  * number of ordinal multinomial outcome categories: $(nsamples(x))")
    print(io, "  * link function: $(typeof(x.link))\n")
    print(io, "  * sample size: $(nsamples(x))")
end

# make our new type implement the interface defined above
nsamples(trait::OrderedMultinomialModel) = size(trait.X, 1)
neffects(trait::OrderedMultinomialModel) = size(trait.X, 2)
ncategories(trait::OrderedMultinomialModel) = length(trait.θ) + 1


"""
VarianceComponentModel
VarianceComponentModel object is one of the two model framework objects. Stores information about the simulation of multiple traits, under the Linear Mixed Model Framework.
"""
struct VarianceComponentModel{matT1, matT, vcT} <: AbstractTraitModel
    X::matT             # design matrix
    β::matT1            # regression coefficients
    μ::matT            # expected value of response
	vc::vcT
    function VarianceComponentModel(X::matT, β::matT1, μ::matT, vc::vcT) where {matT1, matT, vcT}
        # make a new instance
        new{matT, matT1, vcT}(X, β, μ, vc)
    end
	# what if here you had one that took formula and df and then vectors of covariances
end

function VarianceComponentModel(X::AbstractMatrix, β::AbstractMatrix, vc::Vector{vcT}) where vcT
	μ = X * β
	# create a new instance
    VarianceComponentModel(X, β, μ, vc)
end

# building from linear predictor only
function VarianceComponentModel(x::AbstractMatrix, vc::Vector{vcT}) where vcT
	μ = x
	VarianceComponentModel(nothing, nothing, μ, vc)
end


function VarianceComponentModel(X::AbstractMatrix, β::AbstractMatrix, Σ::Tuple, V::Tuple)
  n_traits = size(β, 2)
  n_people = size(X, 1)
  μ	= zeros(n_people, n_traits)
	vc = [VarianceComponent(Σ[i], V[i]) for i in 1:length(V)]
  for i in 1:n_traits
    #calculate the mean vector
    μ[:, i] .+= X*β[:, i]
  end
  VarianceComponentModel(X, β, μ, vc)
end


function VarianceComponentModel(formulas::Vector{String}, df, Σ, V)
  n_traits = length(formulas)
  n_people = size(df)[1]
  μ	= zeros(n_people, n_traits)
	vc = [VarianceComponent(Σ[i], V[i]) for i in 1:length(V)]
  for i in 1:n_traits
    #calculate the mean vector
    μ[:, i] += mean_formula(formulas[i], df)
  end
  return VarianceComponentModel(nothing, nothing, μ, vc)
end

function VarianceComponentModel(formulas::Vector{String}, df, vc::vcT) where vcT
  n_traits = length(formulas)
  n_people = size(df)[1]
  μ = zeros(n_people, n_traits)
  for i in 1:n_traits
    #find the beta and x from mean_formula **todo
	μ[:, i] += mean_formula(formulas[i], df)
  end
  return VarianceComponentModel(nothing, nothing, μ, vc)
end

##  Variance Component Model
function show(io::IO, x::VarianceComponentModel)
    print(io, "Variance Component Model\n")
    print(io, "  * number of traits: $(nsamples(x))\n")
	print(io, "  * number of variance components: $(nvc(x))\n")
	print(io, "  * number of linear predictors: $(neffects(x))\n")
    print(io, "  * sample size: $(nsamples(x))")
end

# make our new type implement the interface defined above
nsamples(trait::VarianceComponentModel) = size(trait.X, 1)
neffects(trait::VarianceComponentModel) = size(trait.X, 2)
nvc(trait::VarianceComponentModel) = length(trait.vc)


# """
# VarianceComponentModel
# VarianceComponentModel object is one of the two model framework objects. Stores information about the simulation of multiple traits, under the Linear Mixed Model Framework.
# """
# struct VarianceComponentModel{T} <: AbstractTraitModel
#   mu::Matrix{Float64}
#   vc::T
#   function VarianceComponentModel(mu, vc::T) where T
#     return(new{T}(mu, vc))
#   end
#
#   function VarianceComponentModel(mu::Matrix{Float64}, vc::T) where T
#     return(new{T}(mu, vc))
#   end
# end
#
# function VarianceComponentModel(formulas::Vector{String}, df, vc::T) where T
#   n_traits = length(formulas)
#   n_people = size(df)[1]
#   mu = zeros(n_people, n_traits)
#   for i in 1:n_traits
#     #calculate the mean vector
#     mu[:, i] += mean_formula(formulas[i], df)
#   end
#   return VarianceComponentModel(mu, vc)
# end
#
# function VarianceComponentModel(formulas::Vector{String}, df, Σ, V)
#   n_traits = length(formulas)
#   n_people = size(df)[1]
#   mu = zeros(n_people, n_traits)
# 	vc = [VarianceComponent(Σ[i], V[i]) for i in 1:length(V)]
#   for i in 1:n_traits
#     #calculate the mean vector
#     mu[:, i] += mean_formula(formulas[i], df)
#   end
#   return VarianceComponentModel(mu, vc)
# end
#
# function VarianceComponentModel(X::AbstractArray{S, 2}, β::Matrix{Float64}, vc::Vector{T}) where {S, T}
#   n_traits = size(β, 2)
#   n_people = size(X, 1)
#   mu = zeros(n_people, n_traits)
#   for i in 1:n_traits
#     #calculate the mean vector
#     mu[:, i] .+= X*β[:, i]
#   end
#   return VarianceComponentModel(mu, vc)
# end
#
# function VarianceComponentModel(X::AbstractArray{S, 2}, β::Matrix{Float64},  Σ, V) where {S, T}
#   n_traits = size(β, 2)
#   n_people = size(X, 1)
#   mu = zeros(n_people, n_traits)
# 	vc = [VarianceComponent(Σ[i], V[i]) for i in 1:length(V)]
#   for i in 1:n_traits
#     #calculate the mean vector
#     mu[:, i] .+= X*β[:, i]
#   end
#   return VarianceComponentModel(mu, vc)
# end
