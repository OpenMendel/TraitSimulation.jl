#since the GLM package uses the Distribution types from the Distibutions package, we use these packages not to simulate from their existing functions but to 
#use them as type dispatchers for the simulate_glm_trait function

#SIMULATE WEIBULL TRAITS
function simulate_glm_trait(μ, dist::ResponseType{Weibull{T}, L}) where {L, T} #by default any
if dist.location < zero(T)
      error("Location parameter cannot be negative for a Poisson distribution!")
    end
 return weibull_deviate.(μ, dist.shape) 
end

#SIMULATE POISSON traits
function simulate_glm_trait(μ, dist::ResponseType{Poisson{T}, L}) where {L, T} #by default any
if dist.location < zero(T)
      error("Location parameter cannot be negative for a Poisson distribution!")
    end
 return poisson_deviate.(μ) 
end

#SIMULATE NORMAL TRAITS
function simulate_glm_trait(μ, dist::ResponseType{Normal{T}, L}) where {L, T} #by default any
if dist.scale < zero(T)
      error("Scale cannot be negative for a normal distribution!")
    end
 return normal_deviate.(μ, dist.scale) # the sigma is coming from dist.scale "ResponseType" and not additionally from ~N(0,1)
end

#SIMULATE Bernoulli TRAITS
function simulate_glm_trait(μ, dist::ResponseType{Bernoulli{T}, L}) where {L, T} #by default any
 return bernoulli_deviate.(μ) 
end

#SIMULATE BINOMIAL TRAITS
function simulate_glm_trait(μ, dist::ResponseType{Binomial{T}, L}) where {L, T} #by default any
if dist.trials < zero(T)
      error("Trials cannot be negative for a binomial distribution!")
    end
 return binomial_deviate.(μ, dist.trials)
end

#SIMULATE GAMMA TRAITS
function simulate_glm_trait(μ, dist::ResponseType{Gamma{T}, L}) where {L, T} #by default any
if dist.shape < zero(T)
      error("Shape cannot be negative for a gamma distribution!")
    end
 return gamma_deviate.(dist.shape, μ)
end

#********SIMULATE INVERSE GAUSSIAN TRAITS
function simulate_glm_trait(μ, dist::ResponseType{InverseGaussian{T}, L}) where {L, T} #by default any
if dist.scale < zero(T)
      error("Scale must be positive for an inverse Gaussian distribution!")
    end
 return inverse_gaussian_deviate.(dist.scale, μ)
end


#########BERNOULLI 
"""Generates a Bernoulli random deviate with success probability p."""

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

function t_deviate(mu, sigma, df)
  x = randn(mu)
  y = 2 * gamma_deviate(df / 2, one(mu))
  return sigma * (x / sqrt(y / df)) + mu
end

"""Generates a Weibull random deviate with scale lambda
and shape alpha."""

function weibull_deviate(lambda, alpha)
  return lambda * (- log(rand()))^(one(lambda) / alpha)
end
