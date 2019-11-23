using DataFrames
using LinearAlgebra
using TraitSimulation
using Random

"""
VarianceComponent
this VarianceComponent type stores A, B , CholA and CholB so we don't have to compute the cholesky decomposition inside the loop.
"""
struct VarianceComponent
	A::Matrix{Float64} # n_traits by n_traits
	B::Matrix{Float64} # n_people by n_people
	CholA::Cholesky{Float64,Array{Float64,2}} # cholesky decomposition of A
	CholB::Cholesky{Float64,Array{Float64,2}} # cholesky decomposition of B

	function VarianceComponent(A, B) #inner constructor given A, B 
		return(new(A, B, cholesky(A), cholesky(B))) # stores these values (this is helpful so we don't have it inside the loop)
	end
end

"""
LMM_trait_simulation
single LMM trait with given evaluated covariance matrix so not a VarianceComponent type
""" 
function LMM_trait_simulation(mu, vc::Matrix{T}) where T
	n_people = size(mu)[1]
	n_traits = size(mu)[2]
	#preallocate memory for the returned dataframe simulated_trait
	simulated_trait = zeros(n_people, n_traits)
	z = Matrix{Float64}(undef, n_people, n_traits)

	#for a single evaluated matrix as the specified covariance matrix
	chol_Σ = cholesky(vc)
	#generate from standard normal
	randn!(z)
	# we want to solve u then v to get the first variane component, v.
	#first matrix vector multiplication using cholesky decomposition

	#need to find which will be CholA, CholB 
	lmul!(chol_Σ.L, z)

	simulated_trait += z

	simulated_trait += mu

	return simulated_trait
end

"""
append_terms
Allows us to append terms to create a VarianceComponent type
""" 
function append_terms!(AB, summand)
	# elements in args are symbols,
	A_esc = esc(summand.args[2])
	B_esc = esc(summand.args[3])
	push!(AB.args, :(VarianceComponent($A_esc, $B_esc)))
end

"""
this vc macro allows us to create a vector of VarianceComponent objects for simulation
"""

macro vc(expression)
	n = length(expression.args)
	# AB is an empty vector of variance components list of symbols
	AB = :(VarianceComponent[]) 
	if expression.args[1] != :+ #if first argument is not plus (only one vc)
		summand = expression 
		append_terms!(AB, summand)
	else #MULTIPLE VARIANCE COMPONENTS if the first argument is a plus (Sigma is a sum multiple variance components)
		for i in 2:n
			summand = expression.args[i]
			append_terms!(AB, summand)
		end
	end
	return(:($AB)) # change this to return a vector of VarianceComponent objects
end 


"""
vcobjectuple(vcobject)
This function creates a tuple of Variance Components, given a vector of variancecomponent objects to be compatible with VarianceComponentModels.jl
"""
function  vcobjtuple(vcobject::Vector{VarianceComponent})
	m = length(vcobject)
	d = size(vcobject[1].A, 1)
	n = size(vcobject[1].B, 1)

	Σ = ntuple(x -> zeros(d, d), m)
	V = ntuple(x -> zeros(n, n), m)

	for i in eachindex(vcobject)
		copyto!(V[i], vcobject[i].B)
		copyto!(Σ[i], vcobject[i].A)
	end
	return(Σ, V)
end

"""
SimulateMVN!(z, vc)
SimulateMVN(n_people, n_traits, vc::VarianceComponent)
For a single Variance Component, algorithm that will transform standard normal distribution. 
SimulateMVN allows us to preallocate n_people by n_traits and rewrite over this matrix to save memory allocation.
"""
function SimulateMN!(z::Matrix, vc::VarianceComponent)
	#for the ith variance component (VC)
	cholA = vc.CholA # grab (not calculate) the stored Cholesky decomposition of n_traits by n_traits variance component matrix
	cholB = vc.CholB # grab (not calculate) the stored Cholesky decomposition of n_people by n_people variance component matrix

	#Generating MVN(0, I) of length n_people*n_traits we will later reshape into a matrix
	randn!(z)

	# we want to solve u then v to get the first variance component, v.
	# first matrix vector multiplication using the cholesky decomposed CholA, CholB above 
	lmul!(cholB.U, z) # z = Bz
	rmul!(z, cholA.L) # z = BzA

	#second matrix vector multiplication using the he cholesky decomposed CholA, CholB above
	rmul!(z, cholA.U) # z = BzAAt
	lmul!(cholB.L, z) #multiply on left and save to simulated_trait z = BtBzAAt

	#add the effects of each variance component
	return(z)
end

#this function will call the SimulateMVN! so that I write over z (reuse memory allocation) for potentially many simulations
function SimulateMN(n_people, n_traits, vc::VarianceComponent)
	#preallocate memory for the returned dataframe simulated_trait once
	Z = Matrix{Float64}(undef, n_people, n_traits)

	# calls the function to apply the cholesky decomposition (we have stored in vc object)
	# and transforms/updates z ~ MN(0, vc)
	SimulateMN!(Z, vc::VarianceComponent)
	#returns the allocated and now transformed z
	return(z)
end
"""
LMM_trait_simulation(mu, vc::VarianceComponent)
For a single Variance Component object, without computing mean from dataframe and formulas
 i.e given an evaluated matrix of means, simulate from LMM.
"""
function LMM_trait_simulation(mu, vc::VarianceComponent)
	n_people = size(mu)[1]
	n_traits = size(mu)[2]

	z = SimulateMN(n_people, n_traits, vc)
	z += mu
	return z
end

# here z, simulated_trait will get updated but vc will not be touched.
## AGGREGATE MULTIPLE VARIANCE COMPONENTS IN LMM TRAIT OBJECT to creat overall variance 
"""
Aggregate_VarianceComponents(z, total_variance, vc)
Update the simulated trait with the effect of each variance component. We note the exclamation is to indicate this function will mutate or override the values that its given.
"""
function Aggregate_VarianceComponents!(Z::Matrix, total_variance, vc::Vector{VarianceComponent})
	for i in 1:length(vc)
		SimulateMN!(Z, vc[i])
		#add the effects of each variance component
		total_variance += Z
	end
	return total_variance
end

"""
LMM_trait_simulation(mu, vc::Vector{VarianceComponent})
For a vector of Variance Component objects, without computing mean from dataframe and formulas i.e given an evaluated matrix of means, simulate from LMM.
"""
function LMM_trait_simulation(mu, vc::Vector{VarianceComponent})
	n_people = size(mu)[1]
	n_traits = size(mu)[2]

	#preallocate memory for the returned dataframe simulated_trait
	simulated_trait = zeros(n_people, n_traits)

	# both Z, z will be updated together
	# I create capital Z for storage of the MVN with the ith variance component , this will be written over to save memory
	Z = Matrix{Float64}(undef, n_people, n_traits)
	
	# here is the vectorized version of the matrix to use for the actual simulation from MVN distribution
	# note we use MVN to ease matrix matrix multiplication for matrix vector multiplication.

	#using the function above to write over the allocated z and simulated_trait
	Aggregate_VarianceComponents!(Z, simulated_trait, vc)
	#for each trait add the mean --> MN(mu, Sigma)
	simulated_trait += mu

	return simulated_trait
end

# using Random
Random.seed!(1234);
n = 1000   # no. observations
d = 2      # dimension of responses
m = 2      # no. variance components
p = 2      # no. covariates

# n-by-p design matrix
X = randn(n, p)

# p-by-d mean component regression coefficient
B = ones(p, d)  

# a tuple of m covariance matrices
V = ntuple(x -> zeros(n, n), m) 
for i = 1:m-1
  Vi = [j ≥ i ? i * (n - j + 1) : j * (n - i + 1) for i in 1:n, j in 1:n]
  copy!(V[i], Vi * Vi')
end
copy!(V[m], Diagonal(ones(n))) # last covarianec matrix is idendity
# a tuple of m d-by-d variance component parameters
Σ = ntuple(x -> zeros(d, d), m) 
for i in 1:m
  Σi = [j ≥ i ? i * (d - j + 1) : j * (d - i + 1) for i in 1:d, j in 1:d]
  copy!(Σ[i], Σi' * Σi)
end

# mn1 = MatrixNormal(m0, V[1], Σ[1])
# mn2 = MatrixNormal(zeros(size(m0)), V[2], Σ[2])
# bigV = V[1] + V[2]
# bigΣ = Σ[1] + Σ[2]

# bigMN = MatrixNormal(m0, bigV, bigΣ)

# function VCM_simulation(n::Int64, d::Int64, m::Int64, p::Int64, X::Matrix{Float64},
# 						 B::Matrix{Float64}, V, Σ)
# 	# n-by-p design matrix
# 	X = randn(n, p)

# 	# p-by-d mean component regression coefficient
# 	B = ones(p, d)  

# 	# a tuple of m covariance matrices
# 	V = ntuple(x -> zeros(n, n), m) 
	
# 	for i = 1:m-1
#   		Vi = randn(n, 50)
#   		copy!(V[i], Vi * Vi')
# 	end
# 	copy!(V[m], Diagonal(ones(n))) # last covarianec matrix is idendity
# 	# a tuple of m d-by-d variance component parameters
# 	Σ = ntuple(x -> zeros(d, d), m) 
	
	
# 	# form overall nd-by-nd covariance matrix Ω
# 	O = zeros(n * d, n * d)
# 	for i = 1:m
# 	  O += kron(Σ[i], V[i])
# 	end
# end 


