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
    μ::vecT2            # expected value of response μ = g^-1(η), g is the link function
    dist::Type{distT}   # univariate, exponential family of distributions
    link::linkT         # link function g(μ) = X*β = η
    function GLMTrait(X::matT, β::vecT1, η::vecT2, μ::vecT2,
		 distribution::D, link::linkT) where {D,linkT,vecT1,vecT2,matT}
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
    η = zeros(size(X, 1), size(β, 2))
    mul!(η, X, β)
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

function GLMTrait(X::AbstractMatrix, β::AbstractVector, G::AbstractMatrix,
	 				γ::AbstractVector, distribution, link)
  n_traits = size(β[:, :], 2)
  n_people = size(X, 1)
  η = zeros(n_people, n_traits)
  non_gen_covariates = zeros(n_people, n_traits)

  X_full = [X G]
  β_full = vcat(β,γ)
  return GLMTrait(X_full, β_full, distribution, link)
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
	function OrderedMultinomialTrait(X::matT, β::vecT1,
		 θ::vecT2, link::linkT)  where {matT, vecT1, vecT2, linkT}
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



"""
VCMTrait
VCMTrait object is a model framework object that stores information about the simulation model
 of multiple traits, under the Variance Component Model Framework.
"""
struct VCMTrait{T <: Real}
	X::Matrix{T}             # all effects
	β::Matrix{T}             # regression coefficients
	G::AbstractMatrix        # genotypes CHANGE TO SNPARRAY
	γ::Matrix{T}             # effect sizes for each snp
	vc::Vector{VarianceComponent}
	μ::Matrix{T}            # evaluated fixed effect
	Z::Matrix{T}
	function VCMTrait(X, β, G, γ, vc, μ)
		T = eltype(μ)
		Z  = Matrix{T}(undef, size(μ))
		new{T}(X, β, G, γ, vc, μ, Z)
	end
end


function VCMTrait(X::Matrix{T}, β::Matrix{T}, G::SnpArray, γ::Matrix{T},
	 vc::Vector{VarianceComponent}) where T <: Real
	n, p, m, d = size(X, 1), size(X, 2), length(vc), size(β, 2)
	# working arrays
	genovec = zeros(Float64, size(G))
	Base.copyto!(genovec, @view(G[:, :]), model = ADDITIVE_MODEL, impute = true)
	μ = Matrix{T}(undef, n, d)
	μ_null = Matrix{T}(undef, n, d)
	LinearAlgebra.mul!(μ_null, X, β)
    LinearAlgebra.mul!(μ, genovec, γ)
 	μ += μ_null
	# constructor
	VCMTrait(X, β, genovec, γ, vc, μ)
end

function VCMTrait(X::Matrix{T}, β::Matrix{T}, G::SnpArray,
	 γ::Matrix{T}, Σ::Vector{Matrix{T}}, V::Vector{Matrix{T}}) where T <: Real
	vc = [VarianceComponent(Σ[i], V[i]) for i in 1:length(V)]
	return VCMTrait(X, β, vc)
end

function VCMTrait(X::Matrix{T}, β::Matrix{T},
	 vc::Vector{VarianceComponent}) where T <: Real
	n, d = size(X, 1), size(β, 2)
	μ = Matrix{T}(undef, n, d)
	mul!(μ, X, β)
	genovec = zeros(Float64, n, 1)
	copyto!(genovec, X[:, end])
	γ = Matrix{T}(undef, 1, d)
	copyto!(γ, β[end, :])
	return VCMTrait(X, β, genovec, γ, vc, μ)
end

function VCMTrait(X::Matrix{T}, β::Matrix{T},
	 Σ::Vector{Matrix{T}}, V::Vector{Matrix{T}}) where T <: Real
	vc = [VarianceComponent(Σ[i], V[i]) for i in 1:length(V)]
	return VCMTrait(X, β, vc)
end


function VCMTrait(formulas::Vector{String}, df::DataFrame,
	 vc::Vector{VarianceComponent})
  d = length(formulas)
  n = size(df)[1]
  μ = zeros(n, d)
  found_covariates = Symbol[]
  for i in 1:d
    #calculate the mean vector
	μ[:, i], found_markers = mean_formula(formulas[i], df)
	union!(found_covariates, found_markers)
  end
  X = Matrix(df[:, found_covariates])
  p = length(found_covariates)
  β = Matrix{Float64}(undef, p, d)
  G = Matrix{Float64}(undef, n, 1)
  γ = Matrix{Float64}(undef, 1, p)
  return VCMTrait(X, β, G, γ, vc, μ)
end

function VCMTrait(formulas::Vector{String}, df::DataFrame,
	 Σ::Vector{Matrix{T}}, V::Vector{Matrix{T}}) where T <: Real
	vc = [VarianceComponent(Σ[i], V[i]) for i in 1:length(V)]
	VCMTrait(formulas, df, vc)
end


#  Variance Component Model
function Base.show(io::IO, x::VCMTrait)
    print(io, "Variance Component Model\n")
    print(io, "  * number of traits: $(ntraits(x))\n")
	print(io, "  * number of variance components: $(nvc(x))\n")
    print(io, "  * sample size: $(nsamplesize(x))")
end

#make our new type implement the interface defined above
nsamplesize(trait::VCMTrait) = size(trait.μ, 1)
ntraits(trait::VCMTrait) = size(trait.vc[1].Σ, 1)
nvc(trait::VCMTrait) = length(trait.vc)
neffects(trait::VCMTrait) = size(trait.X, 2)


struct GLMMTrait{distT, linkT, matT2, T, matT} <: AbstractTraitModel
   X::matT             # all effects
   β::matT2            # regression coefficients
   μ::matT2            # mean of the glmm with random effects
   η::matT
   Z::matT             # place holder for getting glmmm mean for simulation
   vc::Vector{T}
   dist::Type{distT}   # univariate, exponential family of distributions
   link::linkT         # link function g(μ) = X*β
end

"""
GLMMTrait
GLMMTrait object is a model framework that stores information about the simulation of multiple traits, under the Generalized Linear Mixed Model Framework.
"""
function GLMMTrait(X::AbstractMatrix, β::AbstractMatrix, vc::Vector, distribution::D, link::linkT) where {D, linkT, matT2, T, matT}
	distT = Base.typename(typeof(distribution)).wrapper
	Z = zeros(size(X, 1), size(β, 2))
    η = similar(Z)
	mul!(η, X, β)
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
ntraits(trait::GLMMTrait) = size(trait.vc[1].Σ, 1)
nvc(trait::GLMMTrait) = length(trait.vc)
neffects(trait::GLMMTrait) = size(trait.X, 2)
