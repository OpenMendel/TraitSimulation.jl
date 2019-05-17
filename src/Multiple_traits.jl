using DataFrames
using LinearAlgebra
using TraitSimulation
using Random

#this VarianceComponent type stores A, B , CholA and CholB so we don't have to compute the cholesky decomposition inside the loop

struct VarianceComponent
	A::Matrix{Float64} # n_traits by n_traits
	B::Matrix{Float64} # n_people by n_people
	CholA::Cholesky{Float64,Array{Float64,2}} # cholesky decomposition of A
	CholB::Cholesky{Float64,Array{Float64,2}} # cholesky decomposition of B

	function VarianceComponent(A, B) #inner constructor given A, B 
		return(new(A, B, cholesky(A), cholesky(B))) # stores these values (this is helpful so we don't have it inside the loop)
	end
end


#multiple LMM traits

#without computing mean from dataframe and formulas i.e given an evaluated matrix of means
function LMM_trait_simulation(mu, vc::Vector{VarianceComponent})
	n_people = size(mu)[1]
	n_traits = size(mu)[2]

	#preallocate memory for the returned dataframe simulated_trait
	simulated_trait = zeros(n_people, n_traits)
	z = Matrix{Float64}(undef, n_people, n_traits)

	for i in 1:length(vc)
		#for the ith variance component (VC)
		cholA = vc[i].CholA
		cholB = vc[i].CholB 

		#Generating MN(0, Sigma)
		# first generate from standard normal
		randn!(z)

		# we want to solve u then v to get the first variance component, v.
		# first matrix vector multiplication using cholesky decomposition

		#need to find which will be CholA, CholB 
		lmul!(cholB.U, z)
		rmul!(z, cholA.L)

		#second matrix vector mult
		rmul!(z, cholA.U)
		lmul!(cholB.L, z) #multiply on left and save to simulated_trait

		#add the effects of each variance component
		simulated_trait += z
	end

	#for each trait add the mean --> MN(mu, Sigma)
	simulated_trait += mu

	out = DataFrame(simulated_trait)

	out = names!(out, [Symbol("trait$i") for i in 1:n_traits])

	return out
end

#single LMM trait 
function LMM_trait_simulation(mu, vc::Matrix{T}) where T
	n_people = size(mu)[1]
	n_traits = size(mu)[2]
	#preallocate memory for the returned dataframe simulated_trait
	simulated_trait = zeros(n_people, n_traits)
	z = Matrix{Float64}(undef, n_people, n_traits)

	chol_Σ = cholesky(vc)
	#generate from standard normal
	randn!(z)

	# we want to solve u then v to get the first variane component, v.
	#first matrix vector multiplication using cholesky decomposition

	#need to find which will be CholA, CholB 
	lmul!(chol_Σ.L, z)

	simulated_trait += z

	#for each trait
	simulated_trait += mu

	out = DataFrame(simulated_trait)

	out = names!(out, [Symbol("trait$i") for i in 1:n_traits])

	return out
end

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
this is a test for vcobjtuple that is compatible with VarianceComponentModels.jl
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