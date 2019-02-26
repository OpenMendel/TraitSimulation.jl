using Distributions
using DataFrames
using LinearAlgebra
using TraitSimulation

#in context of linear mixed models and multiple correlated traits, the outcome must be multivariate normal with 
#GRM and environmental variance matrix or constant (iid) B
 
function multiple_trait_simulation(formulas, dataframe, A, B, GRM)
# for multiple traits
#find the number of traits
n_traits = length(formulas)

mean = Vector{Vector{Float64}}(undef, n_traits)
#for each trait
for i in 1:n_traits
	#calculate the mean vector
	mean[i] = mean_formula(formulas[i], dataframe)
end

#concatenate them together
meanvector = vcat(mean...) # take all of the i's and splat them into the meanvector.

#A = cov(hcat(mean...)) # me assuming that we dont know this A and I have to assume it internally
term1 = kron(A, GRM) 
term2 = kron(B, Matrix{Float64}(I, size(GRM)))
Σ = term1 + term2

model = MvNormal(meanvector, Σ)
out1 = rand(model)

out2 = DataFrame(reshape(out1, (size(GRM)[1], n_traits)))

out2 = names!(out2, [Symbol("trait$i") for i in 1:n_traits])

return out2

end 


####
#version 2 


function multiple_trait_simulation2(formulas, dataframe, A, B, GRM)
	isposdef(GRM)
	isposdef(A)
	isposdef(B)

	#if not then exit and return error ("not semi positive definite")
	#cholesky decomp for A, GRM, B 
	n_people = size(GRM)[1]
	n_traits = size(A)[1]

	cholA = cholesky(A)
	cholK = cholesky(GRM)
	cholB = cholesky(B)

	chol_AK = kron(cholA.L, cholK.L)
	chol_BI = kron(cholB.L, Diagonal(ones(n_people)))

	#generate from standard normal
	z_1 = randn(n_people*n_traits)

# we want to solve u then v to get the first variane component, v.
#first matrix vector multiplication using cholesky decomposition
	u_1 = chol_AK' * z_1

#second matrix vector mult
	v_1 = chol_AK * u_1

	#generate from standard normal
	z_2 = randn(n_people*n_traits)

#for second variance component
	u_2 = chol_BI' * z_2

	v_2 = chol_BI * u_2


simulated_trait = reshape(v_1 + v_2, (n_people, n_traits))

#now that we have simulated from mvn(0, Sigma)
#we need to add back the mean

mean = Matrix{Float64}(undef, n_people, n_traits)
#for each trait
for i in 1:n_traits
	#calculate the mean vector
	mean[:, i] = mean_formula(formulas[i], dataframe)
end

simulated_trait += mean

out = DataFrame(simulated_trait)

out = names!(out, [Symbol("trait$i") for i in 1:n_traits])

return out

end

##

#version 3

function multiple_trait_simulation3(formulas, dataframe, A, B, GRM)
	isposdef(GRM)
	isposdef(A)
	isposdef(B)

	#if not then exit and return error ("not semi positive definite")
	#cholesky decomp for A, GRM, B 
	n_people = size(GRM)[1]
	n_traits = size(A)[1]

	cholA = cholesky(A)
	cholK = cholesky(GRM)
	cholB = cholesky(B)

	#chol_AK = kron(cholA.L, cholK.L)
	#chol_BI = kron(cholB.L, Diagonal(ones(n_people)))

	#generate from standard normal
	z_1 = randn(n_people, n_traits)

# we want to solve u then v to get the first variane component, v.
#first matrix vector multiplication using cholesky decomposition
	#u_1 = chol_AK' * z_1
	new_u1 = cholK.U * z_1 * cholA.L

#second matrix vector mult
	#v_1 = chol_AK * u_1
	new_v1 = cholK.L * new_u1 * cholA.U

	#generate from standard normal
	z_2 = randn(n_people, n_traits)

#for second variance component
	#u_2 = chol_BI' * z_2
	new_u2 = z_2 * cholB.U #identity goes away

	#v_2 = chol_BI * u_2
	new_v2 = new_u2 * cholB.L #identity goes away

#simulated_trait = reshape(new_v1 + new_v2, (n_people, n_traits))
simulated_trait = new_v1 + new_v2

#now that we have simulated from mvn(0, Sigma)
#we need to add back the mean

mean = Matrix{Float64}(undef, n_people, n_traits)
#for each trait
for i in 1:n_traits
	#calculate the mean vector
	mean[:, i] = mean_formula(formulas[i], dataframe)
end

simulated_trait += mean

out = DataFrame(simulated_trait)

out = names!(out, [Symbol("trait$i") for i in 1:n_traits])

return out

end

#version 4 tryiign to make memory allocation better by overwriting 
#variance component matrix 

function multiple_trait_simulation4(formulas, dataframe, A, B, GRM)
	isposdef(GRM)
	isposdef(A)
	isposdef(B)

	#if not then exit and return error ("not semi positive definite")
	#cholesky decomp for A, GRM, B 
	n_people = size(GRM)[1]
	n_traits = size(A)[1]

	cholA = cholesky(A)
	cholK = cholesky(GRM)
	cholB = cholesky(B)

#preallocate memory for the returned dataframe simulated_trait
simulated_trait = zeros(n_people, n_traits)

	#generate from standard normal
	z_1 = randn(n_people, n_traits)

# we want to solve u then v to get the first variane component, v.
#first matrix vector multiplication using cholesky decomposition

	mul!(simulated_trait, cholK.U, z_1)
	rmul!(simulated_trait, cholA.L)

#second matrix vector mult

	rmul!(simulated_trait, cholA.U)
	lmul!(cholK.L, simulated_trait) #multiply on left and save to simulated_trait

	#generate from standard normal
	z_2 = randn(n_people, n_traits)

#for second variance component

	mul!(temp, z_2, cholB.U) 
	rmul!(temp, cholB.L)
	simulated_trait += temp

#for each trait
for i in 1:n_traits
	#calculate the mean vector
	simulated_trait[:, i] += mean_formula(formulas[i], dataframe)
end

out = DataFrame(simulated_trait)

out = names!(out, [Symbol("trait$i") for i in 1:n_traits])

return out

end

## scaling to more than 2 variance components 
# make generic kron(A1, B1) + kron(A2, B2) structure for cov matrices
#version 5
function multiple_trait_simulation5(formulas, dataframe, A, B, GRM)
	isposdef(GRM)
	isposdef(A)
	isposdef(B)

	#if not then exit and return error ("not semi positive definite")
	#cholesky decomp for A, GRM, B 
	n_people = size(GRM)[1]
	n_traits = size(A)[1]

	cholK = cholesky(GRM)

#preallocate memory for the returned dataframe simulated_trait
simulated_trait = zeros(n_people, n_traits)
z = Matrix(undef, n_people, n_traits)

for i in 1:length(vc)

chol_i = cholesky(vc[i])
	cholesky!(cholA, A[i]) #for the ith covariance matrix (VC) in A 
	#if A is a list of matrices 

	cholesky!(cholB, B[i]) #for the ith covariance matrix (VC) in B

	#generate from standard normal

	randn!(z, n_people, n_traits)

# we want to solve u then v to get the first variane component, v.
#first matrix vector multiplication using cholesky decomposition

#need to find which will be CholA, CholB 
	lmul!(cholK.U, z)
	rmul!(z, cholA.L)

#second matrix vector mult

	rmul!(z, cholA.U)
	lmul!(cholK.L, z) #multiply on left and save to simulated_trait

simulated_trait += z
# 	#generate from standard normal
# 	z_2 = randn(n_people, n_traits)

# #for second variance component

# 	mul!(temp, z_2, cholB.U) 
# 	rmul!(temp, cholB.L)
	
end

#for each trait
for i in 1:n_traits
	#calculate the mean vector
	simulated_trait[:, i] += mean_formula(formulas[i], dataframe)
end

out = DataFrame(simulated_trait)

out = names!(out, [Symbol("trait$i") for i in 1:n_traits])

return out

end
#third version using Hua's VCM package to make use of model structure
# hua does not use cholesky decomposition in his kroxaxpy! function



# #test 
# cd /Users/sarahji/Desktop/OpenMendel_Sarah/Tutorials/Heritability 
# snps = SnpArray("heritability.bed")
# minor_allele_frequency = maf(snps)
# common_snps_index = (0.05 .≤ minor_allele_frequency)
# common_snps = SnpArrays.filter("heritability", trues(212), common_snps_index)
# df = convert(Matrix{Float64}, snps)
# df = DataFrame(df)
# GRM = grm(common_snps)
# #B = [0.3 0.1 0.01; 0.1 0.3 0.1; 0.01 0.1 0.3]
# formulaszzz = ["1 + 3(x1)", "1 + 3(x2) + abs(x3)"]
# multiple_trait_simulation3(formulaszzz, df, A, B, GRM)