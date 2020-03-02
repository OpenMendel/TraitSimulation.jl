using GLM, Statistics
import GLM: linkinv, linkfun
import Distributions: UnivariateDistribution

# root type: use this to define an interface shared by
# all instances of AbstractTrait
abstract type AbstractTraitModel end

# for example, you might want to be able to check
# the number of samples or linear predictors

__default_behavior(trait) = error("function not supported by $(typeof(trait))")

"Check the number of observations."
nsamplesize(trait::AbstractTraitModel) = __default_behavior(trait)

"Check the number of linear predictors."
neffects(trait::AbstractTraitModel) = __default_behavior(trait)

"Check the number of variance components."
nvc(trait::AbstractTraitModel) = __default_behavior(trait)

"Simulate the trait under the given model."
simulate(trait::AbstractTraitModel) = __default_behavior(trait)

"Simulate a trait `n` times independently."
simulate(trait::AbstractTraitModel, n::Integer) = __default_behavior(trait)

"Simulate a trait and store the result in y."
simulate!(y, trait::AbstractTraitModel) = __default_behavior(trait)


# Now let's define our first concrete type.
struct GLMTrait{distT, linkT, vecT1, vecT2, matT} <: AbstractTraitModel
    X::matT             # all effects
    β::vecT1            # regression coefficients
    η::vecT2            # linear predictor η = X*β
    μ::vecT2            # expected value of response μ = g^-1(η)  where g is the link function
    dist::Type{distT}   # univariate, exponential family of distributions
    link::linkT         # link function g(μ) = X*β = η
    function GLMTrait(X::matT, β::vecT1, η::vecT2, μ::vecT2, distribution::D, link::linkT) where {D,linkT,vecT1,vecT2,matT}
        # extract the base type without type parameters
        distT = Base.typename(typeof(distribution)).wrapper
        # make a new instance
        new{distT, linkT, vecT1, vecT2, matT}(X, β, η, μ, distT, link)
    end
end

# define outer constructors that act as intermediates between the internal
# constructor and any external interfaces we deem convenient

# building from model encoded as mat-vec
function GLMTrait(X::AbstractMatrix, β::AbstractVector, distribution, link)
    # define the linear predictor
    η = X * β
    # apply the inverse link element-wise
    μ = linkinv.(link, η)
    # create a new instance
    GLMTrait(X, β, η, μ, distribution, link)
end

# building from linear predictor only
function GLMTrait(x::AbstractVector, distribution, link, ismu::Bool = true)
    if ismu
        μ = x
        η = linkfun.(link, μ)
    else
        η = x
        μ = linkinv.(link, η)
    end
    return GLMTrait(nothing, nothing, η, μ, distribution, link)
end

# better printing; customize how a type is summarized in a REPL
function Base.show(io::IO, x::GLMTrait)
    print(io, "Generalized Linear Model\n")
    print(io, "  * response distribution: $(x.dist)\n")
    print(io, "  * link function: $(typeof(x.link))\n")
    print(io, "  * sample size: $(nsamplesize(x))")
end

# make our new type implement the interface defined above
nsamplesize(trait::GLMTrait) = length(trait.μ)
neffects(trait::GLMTrait) = size(trait.X, 2)

struct OrderedMultinomialTrait{matT, vecT1, vecT2, linkT} <: AbstractTraitModel
    X::matT             # all effects
    β::vecT1            # regression coefficients
    θ::vecT2            # must be increasing
	link::linkT	        # link function from GLM.jl
	# create a new instance
	function OrderedMultinomialTrait(X::matT, β::vecT1, θ::vecT2, link::linkT)  where {matT, vecT1, vecT2, linkT}
    return new{matT, vecT1, vecT2, linkT}(X, β, θ, link)
  end
end

function Base.show(io::IO, x::OrderedMultinomialTrait)
    print(io, "Ordinal Multinomial Model\n")
    print(io, "  * number of fixed effects: $(neffects(x))\n")
	print(io, "  * number of ordinal multinomial outcome categories: $(noutcomecategories(x))\n")
    print(io, "  * link function: $(typeof(x.link))\n")
    print(io, "  * sample size: $(nsamplesize(x))")
end

# make our new type implement the interface defined above
nsamplesize(trait::OrderedMultinomialTrait) = size(trait.X, 1)
neffects(trait::OrderedMultinomialTrait) = size(trait.X, 2)
noutcomecategories(trait::OrderedMultinomialTrait) = length(trait.θ) + 1

# variance component models: multiple correlated traits
"""
VCMTrait
VCMTrait object is one of the two model framework objects. Stores information about the simulation of multiple traits, under the Variance Component Model Framework.
"""
struct VCMTrait{matT1, matT2, matT3, T} <: AbstractTraitModel
	X::matT1             # all effects
    β::matT2            # regression coefficients
	mu::matT3
	vc::Vector{T}
	function VCMTrait(X::matT1, β::matT2, mu::matT3, vc::Vector{T}) where {matT1, matT2, matT3, T}
		return new{matT1, matT2, matT3, T}(X, β, mu, vc)
	end
end

function VCMTrait(mu::muT, Ω::AbstractMatrix) where muT
	vc = TotalVarianceComponent(Ω)
  return VCMTrait(nothing, nothing, mu, [vc])
end

function VCMTrait(formulas::Vector{String}, df::DataFrame, vc::T) where T
  n_traits = length(formulas)
  n_people = size(df)[1]
  mu = zeros(n_people, n_traits)
  found_covariates = Symbol[]
  for i in 1:n_traits
    #calculate the mean vector
	mu[:, i], found_markers = mean_formula(formulas[i], df)
	union!(found_covariates, found_markers)
  end
  X = df[:, found_covariates]
  return VCMTrait(X, nothing, mu, vc)
end

function VCMTrait(formulas::Vector{String}, df::DataFrame, Σ, V)
  n_traits = length(formulas)
  n_people = size(df)[1]
  mu = zeros(n_people, n_traits)
	vc = [VarianceComponent(Σ[i], V[i]) for i in 1:length(V)]
found_covariates = Symbol[]
  for i in 1:n_traits
    #calculate the mean vector
    mu[:, i], found_markers = mean_formula(formulas[i], df)
	union!(found_covariates, found_markers)
  end
  X = df[:, found_covariates]
  return VCMTrait(X, nothing, mu, vc)
end

function VCMTrait(X::T1, β::AbstractArray, vc::Vector{T}) where {T1, T}
  n_traits = size(β, 2)
  n_people = size(X, 1)
  mu = zeros(n_people, n_traits)
  for i in 1:n_traits
    #calculate the mean vector
    mu[:, i] .+= X*β[:, i]
  end
  return VCMTrait(X, β, mu, vc)
end

function VCMTrait(X::AbstractArray{T1, 2}, β::Matrix{Float64}, Σ::Vector{Matrix{Float64}}, V::Vector{Matrix{Float64}}) where T1
  n_traits = size(β, 2)
  n_people = size(X, 1)
  mu = zeros(n_people, n_traits)
	vc = [VarianceComponent(Σ[i], V[i]) for i in 1:length(V)]
  for i in 1:n_traits
    #calculate the mean vector
    mu[:, i] .+= X*β[:, i]
  end
  return VCMTrait(X, β, mu, vc)
end

##  Variance Component Model
function Base.show(io::IO, x::VCMTrait)
    print(io, "Variance Component Model\n")
    print(io, "  * number of traits: $(ntraits(x))\n")
	print(io, "  * number of variance components: $(nvc(x))\n")
    print(io, "  * sample size: $(nsamplesize(x))")
end

# make our new type implement the interface defined above
nsamplesize(trait::VCMTrait) = size(trait.mu, 1)
ntraits(trait::VCMTrait) = size(trait.mu, 2)
nvc(trait::VCMTrait) = length(trait.vc)
neffects(trait::VCMTrait) = size(trait.X, 2)


struct GLMMTrait{distT, linkT, matT2, T, matT} <: AbstractTraitModel
   X::matT             # all effects
   β::matT2            # regression coefficients
   μ::matT2            # mean of the glmm with random effects
   η::matT
   Z::matT             # place holder for getting glmmm mean
   vc::Vector{T}
   dist::Type{distT}   # univariate, exponential family of distributions
   link::linkT         # link function g(μ) = X*β
end

function GLMMTrait(X::AbstractMatrix, β::AbstractMatrix, vc::Vector, distribution::D, link::linkT) where {D, linkT, matT2, T, matT}
	distT = Base.typename(typeof(distribution)).wrapper
	Z = zeros(size(X, 1), size(β, 2))
	η = X*β
	μ = zero(η)
  return GLMMTrait(X, β, μ, η, Z, vc, distT, link)
end


# better printing; customize how a type is summarized in a REPL
function Base.show(io::IO, x::GLMMTrait)
    print(io, "Generalized Linear Mixed Model\n")
    print(io, "  * response distribution: $(x.dist)\n")
    print(io, "  * link function: $(typeof(x.link))\n")
	print(io, "  * number of variance components: $(nvc(x))\n")
    print(io, "  * sample size: $(nsamplesize(x))")
end
# make our new type implement the interface defined above
nsamplesize(trait::GLMMTrait) = size(trait.μ, 1)
ntraits(trait::GLMMTrait) = size(trait.μ, 2)
nvc(trait::GLMMTrait) = length(trait.vc)
neffects(trait::GLMMTrait) = size(trait.X, 2)
