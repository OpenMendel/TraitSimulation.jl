# supertype of GLM or LMM

abstract type AbstractTrait end

"""
GLMTrait
GLMTrait object is one of the two model framework objects.
GLMTrait stores information about the simulation of a single trait, under the Generalized Linear Model Framework.
GLM.formula store the evaluated formula
GLM.mu stores the untransformed mean
GLM.transmu stores the transformed mean using the canonical inverse link function by default
GLM.link stores the link function, by default canonical link. For Gamma and the NegativeBinomial responde distributions, we use the LogLink() function by default.
GLM.responsedist stores the vector of distributions to run the simulate function on.
"""
struct GLMTrait{MuType, D, T, L<:GLM.Link} <: AbstractTrait
  mu::MuType
  transmu::MuType
  dist::Type{D} # most metttttaaa
  link::L
  responsedist::T # second most meta type

  function GLMTrait(mu, dist::D, link::L) where {D, L}
    transmu = GLM.linkinv.(link, mu)
    responsedist = buildresponsedist.(dist, mu, transmu)
    return(new{typeof(mu), dist, typeof(responsedist), L}(mu, transmu, dist, link, responsedist))
  end
end

function GLMTrait(mu, dist::D; link = canonicallink(dist())) where D
  if (dist == NegativeBinomial || dist == Gamma)
    link = GLM.LogLink()
  end
  return(GLMTrait(mu, dist, link))
end

function buildresponsedist(dist::Type{NegativeBinomial}, mu, transmu)
  r = 1
  μ = 1 / (1 + transmu / r)
  responsedist =  dist(r, μ)
  return(responsedist)
end

function buildresponsedist(dist::Type{Gamma}, mu, transmu)
  r = 1
  μ = 1 / (1 + transmu / r)
  responsedist = dist(r, μ)
  return(responsedist)
end

function buildresponsedist(dist::D, mu, transmu) where D
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

# theta needs to be in increasing order

# lmm: multiple traits
"""
LMMTrait
LMMTrait object is one of the two model framework objects. Stores information about the simulation of multiple traits, under the Linear Mixed Model Framework.
"""
struct LMMTrait{T} <: AbstractTrait
  formulas::Vector{String}
  mu::Matrix{Float64}
  vc::T
  function LMMTrait(mu, vc::T) where T
    return(new{T}(String[], mu, vc))
  end

  function LMMTrait(formulas::Vector{String}, mu::Matrix{Float64}, vc::T) where T
    return(new{T}(formulas, mu, vc))
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
  return(LMMTrait(formulas, mu, vc))
end
# ## given evaluated mean matrix
# function LMMTrait(mu::AbstractArray{T, 2}, vc::U) where {T, U}
#     return(LMMTrait{U}(string.(mu), mu, vc))
# end
