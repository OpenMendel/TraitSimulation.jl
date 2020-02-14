# For a single Variance Component, algorithm that will transform standard normal distribution
function simulate_matrix_normal!(Z::Matrix, vc::VarianceComponent)
	cholA = vc.CholA # grab (not calculate) the stored Cholesky decomposition of n_traits by n_traits variance component matrix
	cholB = vc.CholB # grab (not calculate) the stored Cholesky decomposition of n_people by n_people variance component matrix
	randn!(Z)
	lmul!(cholB.L, Z) # Z => (CholB.L)Z
	rmul!(Z, cholA.U) # Z => (CholB.L)Z(CholA.U) so each Y_i = Z ~ MN(0, A_i = (CholB.L)(CholB.L)^T, B_i = (CholA.U)^T(CholA.U)), i in 1:m
	return(Z) #adds onto Z the effects of each variance component
end

function simulate_matrix_normal(n_people, n_traits, vc::VarianceComponent)
	Z = Matrix{Float64}(undef, n_people, n_traits)
	simulate_matrix_normal!(Z, vc::VarianceComponent) # calls the function to apply the cholesky decomposition (we have stored in vc object)and transforms/updates z ~ MN(0, vc)
	return(Z) #returns the allocated and now transformed Z
end

"""
	VCM_trait_simulation(mu, vc::VarianceComponent)
VCM_trait_simulation can take either specification for the the mean matrix mu and variance components. This is what the simulate() function calls for the VCMTrait object input.
The user can use either the evaluated mean vector and VarianceComponent type vc, or give the design matrix X and coefficient vector beta and the variance components can a set of tuples.
"""
function VCM_trait_simulation(mu, vc::VarianceComponent)
	n_people = size(mu)[1]
	n_traits = size(mu)[2]
	Z = simulate_matrix_normal(n_people, n_traits, vc)
	Z += mu
	return Z
end
"""
	VCM_trait_simulation(mu, vc::AbstractArray{T, 2}) where T
For an evaluated mean matrix and the evaluated covariance matrix (not a VarianceComponent object), simulate from VCM.
"""
function VCM_trait_simulation(mu, vc::AbstractArray{T, 2}) where T # for an evaluated matrix
	n_people = size(mu)[1]
	n_traits = size(mu)[2]
	simulated_trait = zeros(n_people, n_traits)
	z = Matrix{Float64}(undef, n_people, n_traits)
	chol_Σ = cholesky(Symmetric(vc)) #for a single evaluated matrix as the specified covariance matrix
	randn!(z) #generate from standard normal
	lmul!(chol_Σ.L, z)
	simulated_trait += z
	simulated_trait += mu
	return simulated_trait
end

"""
VCM_trait_simulation(mu, vc::Vector{VarianceComponent})
For a vector of Variance Component objects, without computing mean from dataframe and formulas i.e given an evaluated matrix of means, simulate from VCM.
"""
function VCM_trait_simulation(mu::Matrix, vc::Vector{VarianceComponent})
	n_people = size(mu)[1]
	n_traits = size(mu)[2]
	simulated_trait = zeros(n_people, n_traits) #preallocate memory for MVN np x 1 vector later to be reshaped into matrix n x p
	Z = Matrix{Float64}(undef, n_people, n_traits)
	aggregate_variance_components!(Z, simulated_trait, vc) # sum up the m independent, np x 1 vectors, Y = sum( Yi ~ MVN(0, A_i ⊗ B_i) , i in 1:m)
	simulated_trait += mu # add the mean matrix
	return simulated_trait
end

# Given tuples for the variance components
function VCM_trait_simulation(X::AbstractArray{T, 2}, β::Matrix{Float64}, Σ::Tuple, V::Tuple) where T
	n, p = size(X)
	m = length(V)
	d = size(Σ, 1)
	vc = [VarianceComponent(Σ[i], V[i]) for i in 1:length(V)]
	mean = X*β
	vcm_model = VCMTrait(mean, vc)
	vcm_trait = simulate(vcm_model)
	return vcm_trait
end

function VCM_trait_simulation(X::AbstractArray{T, 2}, β::AbstractArray{Float64}, vc::Vector{VarianceComponent}) where T
	mean = X*β
	vcm_model = VCMTrait(mean, vc)
	vcm_trait = simulate(vcm_model)
	return vcm_trait
end
