#since the GLM package uses the Distribution types from the Distibutions package, we use these packages not to simulate from their existing functions but to 
#use them as type dispatchers for the simulate_glm_trait function

# This super type of all response distribution types
abstract type ResponseDistribution end

# a type alias for a single distribution and a vector of distributions
const ResponseDistributionType = Union{ResponseDistribution, Vector{ResponseDistribution}}

#SIMULATE STUDENT-T distributed TRAITS needs to be edited 
#T distribution with v degrees of freedom 
struct TResponse <: ResponseDistribution
df::Int64
scale::Float64
end

function simulate_glm_trait(μ, dist::ResponseDistribution{TResponse})
if dist.location < zero(T)
      error("Location parameter cannot be negative for a Poisson distribution!")
    end
 return t_deviate.(μ, dist.scale, dist.df)
end

#SIMULATE WEIBULL TRAITS
#weibull distributed reponse with degrees of freedom df 
struct WeibullResponse <: ResponseDistribution
df::Float64
end

function simulate_glm_trait(μ, dist::ResponseDistribution{WeibullResponse})
if dist.location < zero(T)
      error("Location parameter cannot be negative for a Poisson distribution!")
    end
 return weibull_deviate.(μ, dist.shape) 
end


#SIMULATE POISSON traits
struct PoissonResponse <: ResponseDistribution
end

function simulate_glm_trait(μ, dist::ResponseDistribution{PoissonResponse)
if dist.location < zero(T)
      error("Location parameter cannot be negative for a Poisson distribution!")
    end
 return poisson_deviate.(μ)  
end

#SIMULATE NORMAL TRAITS
# Normal response type with standard deviation σ
struct NormalResponse <: ResponseDistribution
scale::Float{64}
end

function simulate_glm_trait(μ, dist::ResponseDistribution{NormalResponse})
if dist.scale < zero(T)
      error("Scale cannot be negative for a normal distribution!")
    end
 return normal_deviate.(μ, dist.scale) # the sigma is coming from dist.scale "ResponseType" and not additionally from ~N(0,1)
end

#SIMULATE Bernoulli TRAITS
struct BernoulliResponse <: ResponseDistribution
end

function simulate_glm_trait(μ, dist::ResponseDistribution{BernoulliResponse})
 return bernoulli_deviate.(μ) 
end


#SIMULATE BINOMIAL TRAITS
struct BinomialResponse <: ResponseDistribution
trials::Int64
end

function simulate_glm_trait(μ, dist::ResponseDistribution{BinomialResponse})
if dist.trials < zero(T)
      error("Trials cannot be negative for a binomial distribution!")
    end
 return binomial_deviate.(μ, dist.trials)
end

#SIMULATE GAMMA TRAITS
#simulate from gamma response with shape parameter
struct GammaResponse <: ResponseDistribution
shape::Float64
function GammaResponse(shape::Float64)
  if(shape <= 0)
    error("shape must be greater than zero in a Gamma distribution!")
  else
    new(shape)
  end
end
end

function simulate_glm_trait(μ, dist::ResponseDistribution{GammaResponse})
if dist.shape < zero(T)
      error("Shape cannot be negative for a gamma distribution!")
    end
 return gamma_deviate.(dist.shape, μ)
end

# Inverse Gaussian with shape parameter 
struct InverseGaussianResponse <: ResponseDistribution
shape::Float64
function InverseGaussianResponse(shape)
  if(shape <= 0)
    error("Shape parameter must be greater than zero in a inverse Gaussian distribution!")
  else
    return(new(shape))
  end
end
end

#********SIMULATE INVERSE GAUSSIAN TRAITS
function simulate_glm_trait(μ, dist::ResponseDistribution{InverseGaussianResponse})
if dist.shape < zero(T)
      error("Shape must be positive for an inverse Gaussian distribution!")
    end
 return inverse_gaussian_deviate.(dist.shape, μ)
end


#SIMULATE EXPONENTIAL TRAITS ## EDIT THIS GUY 
#simulate from exponential response with shape parameter
struct ExponentialResponse <: ResponseDistribution
scale::Float64
function ExponentialResponse(scale::Float64)
  if(scale <= 0)
    error("scale must be greater than zero in a Exponential distribution!")
  else
    new(scale)
  end
end
end

function simulate_glm_trait(μ, dist::ResponseDistribution{ExponentialResponse})
if dist.scale < zero(T)
      error("Scale cannot be negative for a Exponential distribution!")
    end
 return exponential_deviate.(μ)
end


#########BERNOULLI 
"""Generates a Bernoulli random deviate with success probability p."""
#********SIMULATE BERNOULLI TRAITS

function bernoulli_deviate(p)
  if rand(eltype(p)) <= p
    return 1
  else
    return 0
  end
end

#########BINOMIAL
"""Generates a binomial random deviate with success probability p
and n trials."""

function binomial_deviate(p, n::Int)
  sucesses = 0
  for i = 1:n
    if rand(eltype(p)) <= p
      sucesses = sucesses + 1
    end
  end
  return sucesses
end

"""Generates an exponential random deviate with mean mu."""

function exponential_deviate(mu)
  return -mu * log(rand(eltype(mu)))
end

"""Generates a gamma deviate with shape parameter alpha
and intensity lambda."""

function gamma_deviate(alpha, lambda)
  n = floor(Int, alpha)
  T = eltype(alpha)
  z = - log(prod(rand(T, n)))
  beta = alpha - n
  if beta <= one(T) / 10^6
    y = zero(T)
  elseif beta < one(T) / 10^3
    y = (beta / rand(T))^(one(T) / (one(T) - beta))
  else
    (r, s) = (one(T) / beta, beta - one(T))
    for i = 1:1000
      u = rand(T, 2)
      y = - log(one(T) - u[1]^r)
      if u[2] <= (y / (one(T) - exp(- y)))^s
        exit
      end
    end
  end
  return (z + y) / lambda
end

"""Generates an inverse Gaussian deviate with mean mu and
scale lambda."""

function inverse_gaussian_deviate(lambda, mu)
  w = mu * randn()^2
  c = mu / (2 * lambda)
  x = mu + c * (w - sqrt(w * (4 * lambda + w)))
  p = mu / (mu + x)
  if rand() < p
    return x
  else
    return mu^2 / x
  end
end 

"""Generates a normal random deviate with mean mu and 
standard deviation sigma."""

function normal_deviate(mu, sigma)
  return sigma * randn(eltype(mu)) + mu
end

"""Generates a Poisson random deviate with mean mu."""

function poisson_deviate(mu)
  (x, p, k) = (exp(-mu), one(eltype(mu)), 0)
  while p > x
    k = k + 1
    p = p * rand(eltype(mu))
  end
  return k - 1
end

"""Generates a t random deviate with, location mu, scale sigma,
and degrees of freedom df."""

function t_deviate(mu, scale, df)
  x = randn(mu)
  y = 2 * gamma_deviate(df / 2, one(mu))
  return scale * (x / sqrt(y / df)) + mu
end

"""Generates a Weibull random deviate with scale lambda
and shape alpha."""

function weibull_deviate(lambda, alpha)
  return lambda * (- log(rand()))^(one(lambda) / alpha)
end

