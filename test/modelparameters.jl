using Random, SnpArrays, DataFrames, GLM
using LinearAlgebra
using BenchmarkTools
Random.seed!(1234)

function generateSPDmatrix(n)
	A = rand(n)
	m = 0.5 * (A * A')
	PDmat = m + (n * Diagonal(ones(n)))
end


function generateRandomVCM(n::Int64, p::Int64, d::Int64, m::Int64)
	# n-by-p design matrix
	X = randn(n, p)

	# p-by-d mean component regression coefficient for each trait
	B = hcat(ones(p, 1), rand(p))

	V = ntuple(x -> zeros(n, n), m)
	for i = 1:m-1
	  copy!(V[i], generateSPDmatrix(n))
	end
	copy!(V[end], Diagonal(ones(n))) # last covarianec matrix is identity

	# a tuple of m d-by-d variance component parameters
	Σ = ntuple(x -> zeros(d, d), m)
	for i in 1:m
	  copy!(Σ[i], generateSPDmatrix(d))
	end
	return(X, B, Σ, V)
end

import TraitSimulation: snparray_simulation
n = 10
p = 2
d = 2
m = 2

df = DataFrame(x = repeat([0.0], n), y = repeat([1.0], n))
dist = Normal()
link = IdentityLink()

# test for correct mean formula
formulas = ["x + 5y", "2 + log(y)"]

evaluated_output = [repeat([5.0], n), repeat([2.0], n)]

for i in eachindex(formulas)
  return(@test mean_formula(formulas[i], df)[1] == evaluated_output[i])
end

X, B, Σ, V = generateRandomVCM(n, p, d, m)
test_vcm1 = VCMTrait(X, B, @vc Σ[1] ⊗ V[1] + Σ[2] ⊗ V[2])
test_vcm1_equivalent = VCMTrait(X, B, [Σ...], [V...])
@test test_vcm1_equivalent.vc[1].V == V[1]
@test typeof(test_vcm1.vc[1]) == VarianceComponent

varcomp = @vc Σ[1] ⊗ V[1] + Σ[2] ⊗ V[2]
test_vcm1 = VCMTrait(X, B, varcomp)

@test eltype(varcomp) == VarianceComponent

@test vcobjtuple(varcomp)[1][1] == Σ[1]
