"""
VarianceComponent
this VarianceComponent type stores A, B , CholA and CholB so we don't have to compute the cholesky decomposition inside the loop.
"""
struct VarianceComponent
	Σ::Matrix{Float64} # n_traits by n_traits
	V::Matrix{Float64} # n_people by n_people
	CholΣ::Cholesky{Float64,Array{Float64,2}} # cholesky decomposition of A
	CholV::Cholesky{Float64,Array{Float64,2}} # cholesky decomposition of B
	function VarianceComponent(Σ, V) #inner constructor given A, B
		return new(Σ, V, cholesky(Symmetric(Σ)), cholesky(Symmetric(V))) # stores these values (this is helpful so we don't have it inside the loop)
	end
end

struct TotalVarianceComponent
	Ω::Matrix{Float64}
	cholΩ::Cholesky{Float64,Array{Float64,2}}
	function TotalVarianceComponent(Ω)
		cholΩ = cholesky(Symmetric(Ω))
		return new(Ω, cholΩ)
	end
end

function Base.show(io::IO, x::VarianceComponent)
    print(io, "Variance Components\n")
    print(io, "  * number of traits: $(ntraits(x))\n")
    print(io, "  * sample size: $(nsamplesize(x))")
end

# make our new type implement the interface defined above
nsamplesize(vc::VarianceComponent) = size(vc.V, 1)
ntraits(vc::VarianceComponent) = size(vc.Σ, 1)


# For a single Variance Component, algorithm that will transform standard normal distribution
function simulate_matrix_normal!(Z::Matrix, vc::VarianceComponent)
	cholΣ = vc.CholΣ # grab (not calculate) the stored Cholesky decomposition of n_traits by n_traits variance component matrix
	cholV = vc.CholV # grab (not calculate) the stored Cholesky decomposition of n_people by n_people variance component matrix
	randn!(Z)
	lmul!(cholV.L, Z) # Z => (CholB.L)Z
	rmul!(Z, cholΣ.U) # Z => (CholB.L)Z(CholA.U) so each Y_i = Z ~ MN(0, A_i = (CholB.L)(CholB.L)^T, B_i = (CholA.U)^T(CholA.U)), i in 1:m
	return(Z) #adds onto Z the effects of each variance component
end

function simulate_matrix_normal!(Z::Matrix, vc::TotalVarianceComponent)
	cholΩ = vc.cholΩ # grab (not calculate) the stored Cholesky decomposition of n_traits by n_traits variance component matrix
	randn!(Z)
	lmul!(cholΩ.L, Z) # Z => (CholB.L)Z
	return(Z) #adds onto Z the effects of each variance component
end

"""
	VCM_trait_simulation(mu::Matrix{Float64}, vc::Vector{TotalVarianceComponent})
For an evaluated mean matrix and the evaluated covariance matrix (not a VarianceComponent object), simulate from VCM.
"""
function VCM_trait_simulation(Y::Matrix, mu::Matrix{Float64}, vc) # for an evaluated matrix
	Z = zero(Y)
	for i in eachindex(vc)
		simulate_matrix_normal!(Z, vc[i]) # this step aggregates the variance components by
		@. Y += Z # summing the independent matrix normals to Y, rewriting over Z for each variance component
	end
	@. Y += mu # add the mean back to shift the matrix normal
	return Y
end
