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
struct GLMTrait{D, Dtype, L<:GLM.Link} <: AbstractTrait
  formula::String
  mu::Float64
  transmu::Float64
  dist::Type{D}
  link::L
  responsedist::Vector{Dtype}

  function GLMTrait(formula, mu, dist::D, link::L) where {D, L}
    transmu = GLM.linkinv(link, mu)
    responsedist = buildresponsedist(dist, mu, transmu)
    return(new{D, eltype(responsedist), L}(formula, mu, transmu, dist, link, responsedist))
  end

  function GLMTrait(mu, dist::D, link::L) where {D, L}
    transmu = GLM.linkinv(link, mu)
    responsedist = buildresponsedist(dist, mu, transmu)
    return(new{D, eltype(responsedist), L}("", mu, transmu, dist, link, responsedist))
  end
end

function buildresponsedist(dist::Type{NegativeBinomial}, mu, transmu)
  copyto!(transmu, GLM.LogLink(mu))
  r = 1
  μ = 1 / (1 + transmu / r)
  responsedist =  dist(r, μ)
  return(responsedist)
end

function buildresponsedist(dist::Type{Gamma}, mu, transmu)
  copyto!(transmu, GLM.LogLink(mu))
  r = 1
  μ = 1 / (1 + transmu / r)
  responsedist = dist(r, μ)
  return(responsedist)
end

function buildresponsedist(dist::D, mu, transmu) where D
  responsedist = dist(transmu)
  return(responsedist)
end

# ## given evaluated mean vector
# function GLMTrait(mu::Number, df, dist::D; link = canonicallink(dist())) where D
#     return(GLMTrait(string(mu), repeat([mu], size(df, 1)), dist, link))
# end

# ## given formula and dataframe for mean vector evaluation
# function GLMTrait(formula::String, df, dist::D; link = canonicallink(dist())) where {D, L}
#     mu = mean_formula(formula, df)
#     Simulated_Trait = [rand(dist(i)) for i in μ]
#     return(GLMTrait(formula, mu, dist, link))
# end


# function Multiple_GLMTraits(formulas, df, dist::D; link = canonicallink(dist())) where D
#   vec = [GLMTrait(formulas[i], df, dist, link) for i in 1:length(formulas)] #vector of GLMTrait objects
#   return(vec)
# end

# we put type of the dist vector as Any since we want to allow for any ResponseType{Poisson(), LogLink()}, ResponseType{Normal(), IdentityLink()}
# function Multiple_GLMTraits(formulas::Vector{String}, df::DataFrame, dist::Vector; link = canonicallink.(dist[i]() for i in 1:length(dist)))
#   vec = [GLMTrait(formulas[i], df, dist[i], link[i]) for i in 1:length(formulas)]
#   return(vec)
# end



# lmm: multiple traits (MVN)
"""
LMMTrait
LMMTrait object is one of the two model framework objects. Stores information about the simulation of multiple traits, under the Linear Mixed Model Framework.
"""
struct LMMTrait{T} <: AbstractTrait
  formulas::Vector{String}
  mu::Matrix{Float64}
  vc::T
  function LMMTrait(formulas, df, vc::T) where T
    n_traits = length(formulas)
    n_people = size(df)[1]
    mu = zeros(n_people, n_traits)
    for i in 1:n_traits
      #calculate the mean vector
      mu[:, i] += mean_formula(formulas[i], df)
    end
    return(new{T}(formulas, mu, vc))
  end

  function LMMTrait(mu, vc::T) where T
    return(new{T}(String[],mu, vc))
  end

end

# ## given evaluated mean matrix
# function LMMTrait(mu::AbstractArray{T, 2}, vc::U) where {T, U}
#     return(LMMTrait{U}(string.(mu), mu, vc))
# end