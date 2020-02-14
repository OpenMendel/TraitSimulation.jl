using Random
Random.seed!(1234)
include("benchmarking.jl")
import TraitSimulation: generateRandomVCM, @vc, snparray_simulation
n = 10
p = 2
d = 2
m = 2

df = DataFrame(x = repeat([0], n), y = repeat([1], n))
dist = Normal()
link = IdentityLink()

# test for correct error
@test_throws ErrorException GLMTrait(-1, dist)

# test for correct mean formula
formulas = ["x + 5y", "2 + log(y)"]

evaluated_output = [repeat([5], n), repeat([2], n)]

for i in eachindex(formulas)
  return(@test mean_formula(formulas[i], df) == evaluated_output[i])
end

beta = [1, 5]
glmtrait = GLMTrait(Matrix(df), beta, dist, link)

for i in eachindex(evaluated_output[1])
  return(@test GLMTrait(Matrix(df), beta, dist, link).μ[i] == evaluated_output[1][i])
end

X, B, Σ, V = generateRandomVCM(n, p, d, m)

# traitversion1 = VCMTrait(X, B, Σ, V)
# variancecomponent = @vc Σ[1] ⊗ V[1] + Σ[2] ⊗ V[2]

@test VCMTrait(X, B, Σ, V).vc[1].A - (@vc Σ[1] ⊗ V[1] + Σ[2] ⊗ V[2])[1].A == zeros(2, 2)

#testing for types
maf  = 0.2
nsnps = 10
@test snparray_simulation([maf], nsnps) isa SnpArray
