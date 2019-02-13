module TraitSimulation
using GLM # this already defines some useful distribution and link types
using DataFrames # so we can test it 
using StatsModels # useful distributions
using Distributions #lots more useful distributions

struct ResponseType{D<:Distributions.Distribution, L<:GLM.Link}
  family::D
  inverse_link::L
  location::Float64
  scale::Float64
  shape::Float64
  df::Float64
  trials::Int

# #inner constructor
#   function ResponseType(family::D, inverse_link::L, location, scale, shape, df, trials) where {D, L}
#     if !(D isa Distributions.Distribution) 
#       error("Distribution $(D) is not supported!")
#     end
#     if !(L isa GLM.Link)
#       error("Link $(L) is not supported!")
#     end
#     return new{D, L}(family, inverse_link, location, scale, shape, df, trials) #overriding default types in responsetype 
#   end

end



# The first task is to find the variables to make the mean vector:
find_variables(x) = find_variables!(Symbol[], x) #this is so that we can call the function without the exclamation
function find_variables!(var_names, x::Number) #if the variable name is a number then just return it without doing anything
                return var_names
              end
function find_variables!(var_names, x::Symbol) # if the variable name is a symbol then push it to the list of var_names because its a name of a column
                push!(var_names, x)
              end

function find_variables!(var_names, x::Expr) # if the variable is a expression object then we have to crawl through each argument of the expression
                # safety checking
                if x.head == :call  # check for + symbol bc we are summing linear combinations within each expression argument
                  # pass the remaining expression
        for argument in x.args[2:end] # since the first argument is the :+ call 
                    find_variables!(var_names, argument) #recursively find the names given in each argument so check if number, if symbol, if expression etc.again agian
                  end
                end
                return var_names
              end

function search_variables!(x::Expr, var::Symbol)
    for i in eachindex(x.args)
        if x.args[i] == var # if the argument is one of the variables given then just put it in the right format df[:x1] 
            x.args[i] = Meta.parse(string(:input_data_from_user,"[", ":", var, "]"))
                elseif x.args[i] isa Expr # else if the argument is an expression (i.e not a varaible (symbol) or a number) then 
            search_variables!(x.args[i], var) #go through this function recursively on each of the arguments of the expression object
        end
    end
    return x
end

function search_variables!(x::Expr, vars...) # this is for when you have more than one variable name found in the string
    for var in vars #goes through each of the variables in the vector vars
        x = search_variables!(x, var) #runs the recursion on each variable in vars
    end
    return x 
end

function mean_formula(user_formula_string::String, df::DataFrame)
global input_data_from_user = df
    
users_formula_expression = Meta.parse(user_formula_string)

found_markers = find_variables(users_formula_expression) #store the vector of symbols of the found variables 

dotted_args = map(Base.Broadcast.__dot__, users_formula_expression.args) # adds dots to the arguments in the expression 
dotted_expression = Expr(:., dotted_args[1], Expr(:tuple, dotted_args[2:end]...)) #reformats the exprssion arguments by changing the variable names to tuples of the variable names to keep the dot structure of julia

julia_interpretable_expression = search_variables!(dotted_expression, found_markers...) #gives me the julia interpretable exprsesion with the dataframe provided

mean_vector = eval(Meta.parse(string(julia_interpretable_expression))) #evaluates the julia interpretable expression on the dataframe provided
    return mean_vector
end


#not in glm package say weibull assuming i find out what the weibull link is 
#apply_inverse_link(μ, dist::ResponseType{D, LogLink}) where D = weibull_link.(μ)

#in glm package

#apply_inverse_link(μ, dist::ResponseType{D, Any}) where D = error("Link function not supported!")

apply_inverse_link(μ, dist::ResponseType{D, LogLink}) where D = log_inverse_link.(μ)

apply_inverse_link(μ, dist::ResponseType{D, IdentityLink}) where D = identity_inverse_link.(μ)

apply_inverse_link(μ, dist::ResponseType{D, SqrtLink}) where D = sqrt_inverse_link.(μ)

apply_inverse_link(μ, dist::ResponseType{D, ProbitLink}) where D = probit_inverse_link.(μ)

apply_inverse_link(μ, dist::ResponseType{D, LogitLink}) where D = logit_inverse_link.(μ)

apply_inverse_link(μ, dist::ResponseType{D, InverseLink}) where D = inverse_inverse_link.(μ)

apply_inverse_link(μ, dist::ResponseType{D, CauchitLink}) where D = cauchit_inverse_link.(μ)

apply_inverse_link(μ, dist::ResponseType{D, CloglogLink}) where D = cloglog_inverse_link.(μ)

export LogLink, IdentityLink, SqrtLink, ProbitLink, LogitLink, InverseLink, CauchitLink, CloglogLink

function simulate_glm_trait(μ, dist::ResponseType{Weibull{T}, L}) where {L, T} #by default any
if dist.location < zero(T)
      error("Location parameter cannot be negative for a Poisson distribution!")
    end
 return weibull_deviate.(μ, dist.shape) 
end

function simulate_glm_trait(μ, dist::ResponseType{Poisson{T}, L}) where {L, T} #by default any
if dist.location < zero(T)
      error("Location parameter cannot be negative for a Poisson distribution!")
    end
 return poisson_deviate.(μ) 
end

#NORMAL TRAITS
function simulate_glm_trait(μ, dist::ResponseType{Normal{T}, L}) where {L, T} #by default any
if dist.scale < zero(T)
      error("Scale cannot be negative for a normal distribution!")
    end
 return normal_deviate.(μ, dist.scale)
end

#Bernoulli TRAITS
function simulate_glm_trait(μ, dist::ResponseType{Bernoulli{T}, L}) where {L, T} #by default any
 return bernoulli_deviate.(μ) 
end

#BINOMIAL TRAITS
function simulate_glm_trait(μ, dist::ResponseType{Binomial{T}, L}) where {L, T} #by default any
if dist.trials < zero(T)
      error("Trials cannot be negative for a binomial distribution!")
    end
 return binomial_deviate.(μ, dist.trials)
end

#GAMMA TRAITS
function simulate_glm_trait(μ, dist::ResponseType{Gamma{T}, L}) where {L, T} #by default any
if dist.shape < zero(T)
      error("Shape cannot be negative for a gamma distribution!")
    end
 return gamma_deviate.(dist.shape, μ)
end

#********INVERSE GAUSSIAN TRAITS
function simulate_glm_trait(μ, dist::ResponseType{InverseGaussian{T}, L}) where {L, T} #by default any
if dist.scale < zero(T)
      error("Scale must be positive for an inverse Gaussian distribution!")
    end
 return inverse_gaussian_deviate.(dist.scale, μ)
end

export Poisson, Normal, Binomial, Bernoulli, Gamma, InverseGaussian, TDist, Weibull

function actual_simulation(mu, dist::ResponseType)
	transmu = apply_inverse_link(mu, dist)
	Simulated_Trait = simulate_glm_trait(transmu, dist)
	return(Simulated_Trait)
end

"""Generates a Bernoulli random deviate with success probability p."""

function bernoulli_deviate(p)
  if rand(eltype(p)) <= p
    return 1
  else
    return 0
  end
end

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

##LINK FUNCTIONS##

"""inverse cauchit link."""

function cauchit_inverse_link(x)
  return atan(x) / pi + one(x) / 2
end

"""inverse cloglog link."""

function cloglog_inverse_link(x)
  return one(x) - exp(-exp(x))
end 

"""inverse identity link."""

function identity_inverse_link(x)
  return x
end

"""inverse inverse link."""

function inverse_inverse_link(x)
  return one(x) / x
end

"""inverse logit link."""

function logit_inverse_link(x)
  return one(x) / (one(x) + exp(-x))
end

"""inverse log link."""

function log_inverse_link(x)
  return exp(x)
end

"""inverse probit link."""
 
function probit_inverse_link(x)
  return (one(x) + erf(x / sqrt(2 * one(x)))) / 2
end

"""inverse sqrt link."""

function sqrt_inverse_link(x)
  return x * x
end


export ResponseType, actual_simulation, mean_formula
end # module
