function simulate_glm_trait(μ, dist::ExponentialResponse)
  if any(μ_i -> μ_i < 0, μ)
      error("Rate cannot be negative for an Exponential Distribution!")
    end
  return exponential_deviate.(μ)
end
