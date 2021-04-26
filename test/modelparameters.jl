using Random, SnpArrays, DataFrames, GLM
using LinearAlgebra, Test, TraitSimulation
using BenchmarkTools, Statistics
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

# what happens when there is no variables and just a scalar
formulas2 = ["25", "738"]
@test unique(mean_formula(formulas2[1], df)[1]) == [25]

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

varcomp_onevc = @vc Σ[1] ⊗ V[1]
@test eltype(varcomp_onevc) == VarianceComponent

test_vcm1 = VCMTrait(X, B, varcomp)

# check if the structure is correct
@test eltype(varcomp) == VarianceComponent

# check if returns the appropriate decomposition of the VarianceComponent type
@test vcobjtuple(varcomp)[1][1] == Σ[1]

# test provided simulate coefficients function
x = rand(n)
@test eltype(TraitSimulation.simulate_effect_size(x)) == Float64

effectsizes = rand(n)
our_names = ["sarah"; "janet"; "hua"; "eric"; "ken"; "jenny"; "ben"; "chris"; "juhyun"; "xinkai"]
whats_my_mean_formula = TraitSimulation.FixedEffectTerms(effectsizes, our_names)
data_frame_2 = DataFrame(ones(n, n))
rename!(data_frame_2, Symbol.(our_names))

@test unique(mean_formula(whats_my_mean_formula, data_frame_2)[1])[1] == sum(effectsizes)

@test_throws ErrorException TraitSimulation.__default_behavior(test_vcm1)

test_vcm1_new = VCMTrait(X, B, @vc Σ[1] ⊗ V[1] + Σ[2] ⊗ V[2])
test_vcm1_equiv_new = VCMTrait(X, B, [Σ...], [V...])

##
nsim = 10
using Statistics
Y_new = simulate(test_vcm1_new, nsim)
Y_vecd = zeros(n*d, nsim)

for i in 1:nsim
	Y_vecd[:, i]  = vec(Y_new[i])
end

simulated_mean = Statistics.mean(Y_vecd, dims = 2)

Z_new =  Y_vecd .- simulated_mean

emp_cov = (Z_new * Z_new') * inv(nsim)


true_mu = vec(test_vcm1_new.μ)

true_Ω = zeros(n*d, n*d)
for i = 1:m
  global true_Ω += kron(Σ[i], V[i])
end

vs = diag(true_Ω)

    for i = 1:20
        @test isapprox(simulated_mean[i], true_mu[i], atol=sqrt(vs[i] / nsim) * 8.0)
    end
    for i = 1:20, j = 1:20
        @test isapprox(emp_cov[i,j], true_Ω[i,j], atol=sqrt(vs[i] * vs[j]) * 10.0 / sqrt(nsim))
    end
