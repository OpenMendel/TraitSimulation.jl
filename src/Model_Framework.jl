struct GLMTrait
formula::String
mu::Vector{Float64}
dist:: ResponseType
function GLMTrait(formula, df, dist)
    mu = mean_formula(formula, df)
    return(new(formula, mu, dist))
  end
end

function Multiple_GLMTraits(formulas, df, dist::ResponseType)
  vec = [GLMTrait(formulas[i], df, dist) for i in 1:length(formulas)] #vector of GLMTrait objects
  return(vec)
end

# we put type of the dist vector as Any since we want to allow for any ResponseType{Poisson(), LogLink()}, ResponseType{Normal(), IdentityLink()}
function Multiple_GLMTraits(formulas::Vector{String}, df::DataFrame, dist::Vector)
  vec = [GLMTrait(formulas[i], df, dist[i]) for i in 1:length(formulas)]
  return(vec)
end


# lmm: multiple traits (MVN)

struct LMMTrait
formulas::Vector{String}
mu::Matrix{Float64}
vc::Vector{VarianceComponent}
  function LMMTrait(formulas, df, vc)
    n_traits = length(formulas)
    n_people = size(df)[1]
    mu = zeros(n_people, n_traits)
    for i in 1:n_traits
      #calculate the mean vector
      mu[:, i] += mean_formula(formulas[i], df)
    end
    return(new(formulas, mu, vc))
  end
end