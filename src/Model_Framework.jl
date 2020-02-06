# supertype of GLM or LMM

abstract type AbstractTrait end

"""
GLMTrait
GLMTrait object is one of the two model framework objects.
GLMTrait stores information about the simulation of a single trait, under the Generalized Linear Model Framework.
GLM.formula store the evaluated formula
GLM.X store the matrix of covariates
GLM.β stores the vector of fixed effects
GLM.transmu stores the transformed mean using the canonical inverse link function by default
GLM.link stores the link function, by default canonical link. For Gamma and the NegativeBinomial responde distributions, we use the LogLink() function by default.
GLM.responsedist stores the vector of distributions to run the simulate function on.
"""
struct GLMTrait{C, B, MuType, D, L<:GLM.Link, T} <: AbstractTrait
  X::C
  β::B
  transmu::MuType
  dist::Type{D} # most metttttaaa
  link::L
  responsedist::T # second most meta type

  function GLMTrait(X::Matrix{C}, β::Vector{B}, dist::D, link::L) where {C, B, D, L}
    mu = X*β
    transmu = GLM.linkinv.(link, mu)
    responsedist = buildresponsedist.(dist, transmu)
  return(new{typeof(X), typeof(β), typeof(transmu), dist, L, typeof(responsedist)}(X, β, transmu, dist, link, responsedist))
  end
end


function GLMTrait(X::Matrix{C}, β::Vector{B}, dist::D; link = canonicallink(dist())) where {C, B, D}
  if (dist == NegativeBinomial || dist == Gamma)
    link = GLM.LogLink()
  end
  return(GLMTrait(X, β, dist, link))
end

function buildresponsedist(dist::Type{NegativeBinomial}, transmu)
  r = 1
  μ = 1 / (1 + transmu / r)
  responsedist =  dist(r, μ)
  return(responsedist)
end

function buildresponsedist(dist::Type{Gamma}, transmu)
  r = 1
  μ = 1 / (1 + transmu / r)
  responsedist = dist(r, μ)
  return(responsedist)
end

function buildresponsedist(dist::D, transmu) where D
  responsedist = dist.(transmu)
  return(responsedist)
end

#rpolr(X, β, θ, link)
struct OrdinalTrait{T, L<:GLM.Link} <: AbstractTrait
  X::Matrix{T}
  β::Vector{T}
  θ::Vector{T}
  link::L
  function OrdinalTrait(X::Matrix{T}, β::Vector{T}, θ::Vector{T}, link::L) where {T, L}
    return(new{T, L}(X, β, θ, link))
  end
end

function LMM_trait_simulation(X::AbstractArray{T, 2}, B::Matrix{Float64}, Σ, V) where T
	n, p = size(X)
	m = length(V)
	d = size(Σ, 1)
	VC = [VarianceComponent(Σ[i], V[i]) for i in 1:length(V)]
	mean = X*B
	VCM_Model = LMMTrait(mean, VC)
	VCM_trait = simulate(VCM_Model)
	return(VCM_trait)
end

function LMM_trait_simulation(X::AbstractArray{T, 2}, B::Matrix{Float64}, VC::Vector{VarianceComponent}) where T
	mean = X*B
	VCM_Model = LMMTrait(mean, VC)
	VCM_trait = simulate(VCM_Model)
	return(VCM_trait)
end

# lmm: multiple traits
"""
LMMTrait
LMMTrait object is one of the two model framework objects. Stores information about the simulation of multiple traits, under the Linear Mixed Model Framework.
"""
struct LMMTrait{T} <: AbstractTrait
  mu::Matrix{Float64}
  vc::T
  function LMMTrait(mu, vc::T) where T
    return(new{T}(mu, vc))
  end

  function LMMTrait(mu::Matrix{Float64}, vc::T) where T
    return(new{T}(mu, vc))
  end
end

function LMMTrait(formulas::Vector{String}, df, vc::T) where T
  n_traits = length(formulas)
  n_people = size(df)[1]
  mu = zeros(n_people, n_traits)
  for i in 1:n_traits
    #calculate the mean vector
    mu[:, i] += mean_formula(formulas[i], df)
  end
  return(LMMTrait(mu, vc))
end

function LMMTrait(formulas::Vector{String}, df, Σ, V)
  n_traits = length(formulas)
  n_people = size(df)[1]
  mu = zeros(n_people, n_traits)
	vc = [VarianceComponent(Σ[i], V[i]) for i in 1:length(V)]
  for i in 1:n_traits
    #calculate the mean vector
    mu[:, i] += mean_formula(formulas[i], df)
  end
  return(LMMTrait(mu, vc))
end

function LMMTrait(X::AbstractArray{S, 2}, β::Matrix{Float64}, vc::Vector{T}) where {S, T}
  n_traits = size(β, 2)
  n_people = size(X, 1)
  mu = zeros(n_people, n_traits)
  for i in 1:n_traits
    #calculate the mean vector
    mu[:, i] .+= X*β[:, i]
  end
  return(LMMTrait(mu, vc))
end

function LMMTrait(X::AbstractArray{S, 2}, β::Matrix{Float64},  Σ, V) where {S, T}
  n_traits = size(β, 2)
  n_people = size(X, 1)
  mu = zeros(n_people, n_traits)
	vc = [VarianceComponent(Σ[i], V[i]) for i in 1:length(V)]
  for i in 1:n_traits
    #calculate the mean vector
    mu[:, i] .+= X*β[:, i]
  end
  return(LMMTrait(mu, vc))
end
