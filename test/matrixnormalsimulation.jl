using Random, SnpArrays
using LinearAlgebra
Random.seed!(1234)
import TraitSimulation: snparray_simulation
n = 10
p = 2
d = 2
m = 2

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

X, B, Σ, V = generateRandomVCM(n, p, d, m)
# specify the variance compoents of the model
varcomp = @vc Σ[1] ⊗ V[1] + Σ[2] ⊗ V[2]

test_vcm1 = VCMTrait(X, B, varcomp)

y_vcm = simulate(test_vcm1)

@test size(y_vcm) == size(X*B)
number_independent_simulations = 5
@test length(simulate(test_vcm1, number_independent_simulations)) == number_independent_simulations

@test test_vcm1 isa TraitSimulation.AbstractTraitModel

#testing for types
minor_alle_frequency  = 0.2
nsnps = p
@test snparray_simulation([minor_alle_frequency], nsnps) isa SnpArrays.SnpArray

variance_formula2  = @vc [minor_alle_frequency][:,:] ⊗ V[1] + [minor_alle_frequency][:,:] ⊗ V[1]
trait2 = VCMTrait(["x2 + 3x1", "2 +2x1"], DataFrame(X), variance_formula2)
sigma, v = vcobjtuple(variance_formula2)
trait2_equivalent = VCMTrait(["x2 + 3x1", "2 +2x1"], DataFrame(X), [sigma...], [v...])

@test trait2_equivalent.vc[1].V == trait2.vc[1].V

# simulate the SnpArray
G = snparray_simulation([0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5], n)
γ = rand(7, 2)

vcmOBJ =  VCMTrait(X, B, G, γ, varcomp)
vcmOBJ_equivalent =  VCMTrait(X, B, G, γ, [Σ...], [V...])

@test vcmOBJ.vc[1].V == V[1]
@test vcmOBJ_equivalent.vc[1].V == V[1]
# check data was copied correctly
@test vcmOBJ.X == X

# check that the overall mean is the sum of both nongenetic and genetic effects
@test vcmOBJ.μ == X*B + vcmOBJ.G*γ
#X*β .+ genovec*γ

vcmOBJ2 =  VCMTrait(X, B, varcomp)

@test eltype(simulate(vcmOBJ2)) == Float64

@test length(simulate(vcmOBJ2, number_independent_simulations))  == number_independent_simulations


vcmOBJ2_equivalent = VCMTrait(X, B, [Σ...], [V...])
@test vcmOBJ2.vc[1].V == vcmOBJ2_equivalent.vc[1].V
@test vcmOBJ2.G != nothing
@test vcmOBJ2.μ == X*B

# check if the preallocated memory that will loop over for the matrix normal  (to sum the variance components) before we sum the mean matrix
@test size(vcmOBJ2.Z) == size(vcmOBJ2.μ)


y_alternative = zeros(size(vcmOBJ2.μ))
function testing_allocations(vcmOBJ2, y_alternative)
    TraitSimulation.simulate!(y_alternative, vcmOBJ2)
end
bmrk  = @benchmark testing_allocations(vcmOBJ2, y_alternative)

@test bmrk.memory == 0
@test bmrk.allocs ==  0
