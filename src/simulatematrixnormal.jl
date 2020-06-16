using VarianceComponentModels, Distributions
"""
VarianceComponent
this VarianceComponent type stores A, B , CholA and CholB so we don't have to compute the cholesky decomposition inside the loop.
"""
struct VarianceComponent
	Σ::Matrix{Float64} # n_traits by n_traits
	V::Matrix{Float64} # n_people by n_people
	CholΣ::Array{Float64,2} # cholesky decomposition of A
	CholV::Array{Float64,2} # cholesky decomposition of B
	function VarianceComponent(Σ, V) #inner constructor given A, B
		CholΣ = cholesky(Symmetric(Σ)).factors
		CholV = cholesky(Symmetric(V)).factors
		return new(Σ, V, CholΣ, CholV) # stores these values (this is helpful so we don't have it inside the loop)
	end
end

function Base.show(io::IO, x::VarianceComponent)
    print(io, "Variance Component\n")
    print(io, "  * number of traits: $(ntraits(x))\n")
    print(io, "  * sample size: $(nsamplesize(x))")
end

#make our new type implement the interface defined above
nsamplesize(vc::VarianceComponent) = size(vc.V, 1)
ntraits(vc::VarianceComponent) = size(vc.Σ, 1)

# For a single Variance Component, algorithm that will transform standard normal distribution
function simulate_matrix_normal!(Z::Matrix, vc::VarianceComponent)
	randn!(Z)
	BLAS.trmm!('L', 'U', 'T', 'N', 1.0, vc.CholV, Z)
	BLAS.trmm!('R', 'U', 'N', 'N', 1.0, vc.CholΣ, Z)
	return Z #adds onto Z the effects of each variance component
end

"""
	VCM_trait_simulation(mu::Matrix{Float64}, vc::Vector{VarianceComponent})
For an evaluated mean matrix and vector of VarianceComponent objects, simulate from VCM.
"""
function VCM_trait_simulation(Y::Matrix, Z::Matrix, mu::Matrix{Float64}, vc::Vector{VarianceComponent}) # for an evaluated matrix
	for i in eachindex(vc)
		TraitSimulation.simulate_matrix_normal!(Z, vc[i]) # this step aggregates the variance components by
		axpy!(1.0, Z, Y) # summing the independent matrix normals to Y, rewriting over Z for each variance component
	end
	axpy!(1.0, mu, Y) # add the mean back to shift the matrix normal
	return Y
end
