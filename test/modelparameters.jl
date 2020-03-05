using Random, SnpArrays
Random.seed!(1234)
include("benchmarking.jl")
import TraitSimulation: snparray_simulation
n = 10
p = 2
d = 2
m = 2

df = DataFrame(x = repeat([0], n), y = repeat([1], n))
dist = Normal()
link = IdentityLink()

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

# traitversion1 = VCMTrait(X, B, Σ, V)
# variancecomponent = @vc Σ[1] ⊗ V[1] + Σ[2] ⊗ V[2]

X, B, Σ, V = generateRandomVCM(n, p, d, m)
@test VCMTrait(X, B, Σ, V).vc[1].Σ - (@vc Σ[1] ⊗ V[1] + Σ[2] ⊗ V[2])[1].Σ == zeros(2, 2)

#testing for types
maf  = 0.2
nsnps = 10
@test snparray_simulation([maf], nsnps) isa SnpArrays.SnpArray

using LinearAlgebra
X, β, Σ, V  = generateRandomVCM(50,2 , 2, 2)
G = snparray_simulation([0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5], 50)
Σ = [Σ...]
V = [V...]
γ = rand(7, 2)
varcomp = @vc Σ[1] ⊗ V[1] + Σ[2] ⊗  V[2]
vcmOBJ =  VCMTrait(X, β, G, γ, Σ, V)

@test vcmOBJ.mu == X*β .+ G*γ

vcmOBJ2 =  VCMTrait(X, β, varcomp)
@test isnothing(vcmOBJ2.G)
@test vcmOBJ2.mu ≈ X*β rtol = 0.005

simulations_test  =  simulate(vcmOBJ2, 1000)
using Statistics

using VarianceComponentModels
