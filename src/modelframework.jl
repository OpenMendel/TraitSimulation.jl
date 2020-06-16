using GLM, Statistics
import GLM: linkinv, linkfun
import Distributions: UnivariateDistribution

# root type: use this to define an interface shared by
# all instances of AbstractTrait
abstract type AbstractTraitModel end

__default_behavior(trait) = error("function not supported by $(typeof(trait))")

"Check the number of traits."
ntraits(trait::AbstractTraitModel) = __default_behavior(trait)

"Check the number of variance components."
nvc(trait::AbstractTraitModel) = __default_behavior(trait)

"Check the number of outcome categories."
noutcomecategories(trait::AbstractTraitModel) = __default_behavior(trait)

"Check the number of fixed covariates in linear model."
neffects(trait::AbstractTraitModel) = size(trait.X, 2)

"Check the number of people in the sample."
nsamplesize(trait::AbstractTraitModel) = size(trait.X, 1)

struct GLMTrait{distT, linkT, vecT1, vecT2, vecT3, matT, matT2} <: AbstractTraitModel
	X::matT             # all effects
	β::vecT1            # regression coefficients
	G::matT2        # genotypes CHANGE TO SNPARRAY
	γ::vecT1             # effect sizes for each snp
	η::vecT2            # linear predictor η = X*β
	μ::vecT3            # expected value of response μ = g^-1(η), g is the link function
	dist::Type{distT}   # univariate, exponential family of distributions
	link::linkT         # link function g(μ) = X*β = η
	function GLMTrait(X::matT, β::vecT1, G::matT2, γ::vecT1, η::vecT2, distribution::D, link::linkT; lb = -20, ub = 20) where {D, linkT, vecT1, vecT2, matT, matT2}
	# extract the base type without type parameters
	distT = Base.typename(typeof(distribution)).wrapper
	n = size(X, 1)
	μ = zeros(n)
	for i in eachindex(η)
	μ[i] = GLM.linkinv(link, η[i])
	end
	# make a new instance
	new{distT, linkT, vecT1, typeof(η), typeof(μ), matT, matT2}(X, β, G, γ, η, μ, distT, link)
	end
end

function GLMTrait(X::Matrix{T}, β::Vector{T}, G::SnpArray, γ::Vector{T},
	 distribution::D, link::linkT; lb = -20, ub = 20) where {D, linkT, T <: BlasReal}
	 distT = Base.typename(typeof(distribution)).wrapper
 	 n = size(X, 1)
	 η = Vector{T}(undef, n)
 	 clamp_eta!(η, X, β, distT; lb = lb , ub = ub)
	 genovec = zeros(Float64, size(G))
 	 Base.copyto!(genovec, @view(G[:, :]), model = ADDITIVE_MODEL, impute = true)
  return GLMTrait(X, β, genovec, γ, η, distribution, link)
end

function GLMTrait(X::Matrix{T}, β::Vector{T}, distribution::D, link::linkT; lb = -20, ub = 20) where {D, linkT, T <: BlasReal}
	distT = Base.typename(typeof(distribution)).wrapper
	n = size(X, 1)
	η = zeros(n)
	clamp_eta!(η, X, β, distT; lb = lb , ub = ub)
	genovec = zeros(Float64, n)
	copyto!(genovec, X[:, end])
    GLMTrait(X, β, genovec, [β[end]], η, distribution, link)
end

# clamp the trait values
  # specific to Poisson and NegativeBinomial
  function clamp_eta!(η::AbstractVecOrMat, X::Matrix{T}, β::AbstractVecOrMat,
	   distT::Union{Type{Poisson}, Type{NegativeBinomial}}; lb = -20, ub = 20) where T <: BlasReal
	  mul!(η, X, β)
	  η .= map(y -> y >= ub ? ub : y, η)
  end

  # specific to Bernoulli and Binomial
  function clamp_eta!(η::AbstractVecOrMat, X::Matrix{T}, β::AbstractVecOrMat,
	   distT::Union{Type{Bernoulli}, Type{Binomial}}; lb = -20, ub = 20) where T <: BlasReal
	  mul!(η, X, β)
	  clamp!(η, lb, ub)
	  η
  end

  # all others
  function clamp_eta!(η::AbstractVecOrMat, X::Matrix{T}, β::AbstractVecOrMat,
	   distT::Type{D}; lb = -20, ub = 20) where {D, T <: Real}
	  mul!(η, X, β)
	  η
  end


# better printing; customize how a type is summarized in a REPL
function Base.show(io::IO, x::GLMTrait)
    print(io, "Generalized Linear Model\n")
    print(io, "  * response distribution: $(x.dist)\n")
    print(io, "  * link function: $(typeof(x.link))\n")
    print(io, "  * sample size: $(nsamplesize(x))")
	print(io, "  * fixed effects: $(neffects(x))")
end

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

noutcomecategories(trait::OrderedMultinomialTrait) = length(trait.θ) + 1

"""
VCMTrait
VCMTrait object is a model framework object that stores information about the simulation model
 of multiple traits, under the Variance Component Model Framework.
"""
struct VCMTrait{T <: Real} <: AbstractTraitModel
	X::Matrix{T}             # all effects
	β::Matrix{T}             # regression coefficients
	G::AbstractVecOrMat        # genotypes CHANGE TO SNPARRAY
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
	 vc::Vector{VarianceComponent}) where T <: BlasReal
	n, p, m, d = size(X, 1), size(X, 2), length(vc), size(β, 2)
	genovec = SnpBitMatrix{Float32}(G, model=ADDITIVE_MODEL, center=true, scale=true);
	μ = Matrix{T}(undef, n, 2)
	μ_null = zeros(n, d)
	LinearAlgebra.mul!(μ_null, X, β)
	for j in 1:size(μ, 2)
		mul!(μ[:, j], genovec, γ[:, j]);
	end
	BLAS.axpby!(1.0, μ_null, 1.0, μ)
	# constructor
	VCMTrait(X, β, genovec, γ, vc, μ)
end

function VCMTrait(X::Matrix{T}, β::Matrix{T}, G::SnpArray,
	 γ::Matrix{T}, Σ::Vector{Matrix{T}}, V::Vector{Matrix{T}}) where T <: BlasReal
	vc = [VarianceComponent(Σ[i], V[i]) for i in 1:length(V)]
	return VCMTrait(X, β, vc)
end

function VCMTrait(X::Matrix{T}, β::Matrix{T},
	 vc::Vector{VarianceComponent}) where T <: BlasReal
	n, d = size(X, 1), size(β, 2)
	μ = Matrix{T}(undef, n, d)
	mul!(μ, X, β)
	genovec = zeros(Float32, n, 1)
	copyto!(genovec, X[:, end])
	γ = Matrix{T}(undef, 1, d)
	copyto!(γ, β[end, :])
	return VCMTrait(X, β, genovec, γ, vc, μ)
end

function VCMTrait(X::Matrix{T}, β::Matrix{T},
	 Σ::Vector{Matrix{T}}, V::Vector{Matrix{T}}) where T <: BlasReal
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
  β = Matrix{Float32}(undef, p, d)
  G = Matrix{Float32}(undef, n, 1)
  γ = Matrix{Float32}(undef, 1, p)
  return VCMTrait(X, β, G, γ, vc, μ)
end

function VCMTrait(formulas::Vector{String}, df::DataFrame,
	 Σ::Vector{Matrix{T}}, V::Vector{Matrix{T}}) where T <: BlasReal
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
ntraits(trait::VCMTrait) = size(trait.vc[1].Σ, 1)
nvc(trait::VCMTrait) = length(trait.vc)

"""
GLMMTrait
GLMMTrait object is a model framework that stores information about the simulation of multiple traits, under the Generalized Linear Mixed Model Framework.
"""
struct GLMMTrait{distT, linkT, T <: BlasReal} <: AbstractTraitModel
   X::Matrix{T}             # all effects
   β::Matrix{T}            # regression coefficients
   μ::Matrix{T}            # mean of the glmm with random effects
   η::Matrix{T}
   Z::Matrix{T}             # place holder for aggregating random effects
   Y_vcm::Matrix{T}        # place holder for getting glmmm mean for simulation
   vc::Vector{VarianceComponent}
   dist::Type{distT}   # univariate, exponential family of distributions
   link::linkT         # link function g(μ) = X*β
   function GLMMTrait(X::Matrix{T}, β::Matrix{T}, vc::Vector{VarianceComponent},
   	 distribution::D, link::linkT; lb = -20, ub = 20) where {D, linkT, T<: BlasReal}
   	distT = Base.typename(typeof(distribution)).wrapper
   	Z = zeros(size(X, 1), size(β, 2))
   	Y_vcm = zeros(size(X, 1), size(β, 2))
    η = zeros(size(X, 1), size(β, 2))
   	clamp_eta!(η, X, β, distT; lb = lb , ub = ub)
   	μ = zero(η)
   	for i in eachindex(η)
   		μ[i] = GLM.linkinv(link, η[i])
   	end
     return new{distT, linkT, T}(X, β, μ, η, Z, Y_vcm, vc, distT, link)
   end
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
ntraits(trait::GLMMTrait) = size(trait.vc[1].Σ, 1)
nvc(trait::GLMMTrait) = length(trait.vc)
