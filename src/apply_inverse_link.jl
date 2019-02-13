apply_inverse_link(μ, dist::ResponseType{D, LogLink}) where D = log_inverse_link.(μ)

apply_inverse_link(μ, dist::ResponseType{D, IdentityLink}) where D = identity_inverse_link.(μ)

apply_inverse_link(μ, dist::ResponseType{D, SqrtLink}) where D = sqrt_inverse_link.(μ)

apply_inverse_link(μ, dist::ResponseType{D, ProbitLink}) where D = probit_inverse_link.(μ)

apply_inverse_link(μ, dist::ResponseType{D, LogitLink}) where D = logit_inverse_link.(μ)

apply_inverse_link(μ, dist::ResponseType{D, InverseLink}) where D = inverse_inverse_link.(μ)

apply_inverse_link(μ, dist::ResponseType{D, CauchitLink}) where D = cauchit_inverse_link.(μ)

apply_inverse_link(μ, dist::ResponseType{D, CloglogLink}) where D = cloglog_inverse_link.(μ)


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
