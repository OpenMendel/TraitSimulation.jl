using CuArrays, CUDAdrv
using CuArrays.CURAND

using TraitSimulation, GLM, LinearAlgebra, Random, BenchmarkTools
using SnpArrays

function simulate_no_threads!(Y, trait::VCMTrait)
  fill!(Y, 0.0)
  TraitSimulation.VCM_trait_simulation(Y, trait.Z, trait.μ, trait.vc)
  return Y
end

"""
```
simulate(trait::VCMTrait, n::Integer)
```
This simulates a VCM trait n times under the desired variance component model, specified using the VCMTrait type.
"""
function simulate_no_threads(trait::VCMTrait, n::Integer)
  # pre-allocate output
  Y_n = [zeros(size(trait.μ)) for _ in 1:n] # replicates
  # do the simulation n times, each thread needs its own Z to collect randome effects
  for k in 1:n
      simulate_no_threads!(Y_n[k], trait)
  end
  return Y_n
end

n_sims = 1000

# @time test_cpu_no_threads = simulate_no_threads(trait, n_sims);

### cpu threaded

# @benchmark test_cpu_threads = simulate(trait, n_sims);

### GPU

struct VC_gpu{matT1, matT2, matT3, matT4}
	Σ::matT1 # n_traits by n_traits
	V::matT2 # n_people by n_people
	CholΣ::matT3 # cholesky decomposition of A
	CholV::matT4 # cholesk decomposition of B
	function VC_gpu(Σ, V) #inner constructor given A, B
        Σd = CuArray(Σ)
        Vd = CuArray(V)
        CholΣ = cholesky(Symmetric(Σd)).factors
        CholV = cholesky(Symmetric(Vd)).factors
        new{typeof(Σd), typeof(Vd), typeof(CholΣ), typeof(CholV)}(Σd, Vd, CholΣ, CholV) # stores these values (this is helpful so we don't have it inside the loop)
	end
end

function append_terms_gpu!(AB, summand)
	A_esc = esc(summand.args[2])	# elements in args are symbols,
	B_esc = esc(summand.args[3])
	push!(AB.args, :(VC_gpu($A_esc, $B_esc)))
end

"""
this vc macro allows us to create a vector of VarianceComponent objects for simulation so with_bigfloat_precis, precision::Integer)
so that the user can type out @vc V[1] ⊗ Σ[1] + V[2] ⊗ Σ[2] + .... + V[m] ⊗ Σ[m]
"""
macro vc_gpu(expression)
	n = length(expression.args)
	AB = :(VC_gpu[]) # AB is an empty vector of variance components list of symbols
	if expression.args[1] != :+ #if first argument is not plus (only one vc)
		summand = expression
		append_terms_gpu!(AB, summand)
	else #MULTIPLE VARIANCE COMPONENTS if the first argument is a plus (Sigma is a sum multiple variance components)
		for i in 2:n
			summand = expression.args[i]
			append_terms_gpu!(AB, summand)
		end
	end
	return(:($AB))
end
    

function simulate_matrix_normal_gpu!(Z_d, vc_gpu)
    randn!(Z_d)
    CuArrays.CUBLAS.trmm!('L', 'U', 'T', 'N', 1.0, vc_gpu.CholV, Z_d, similar(Z_d))
    CuArrays.CUBLAS.trmm!('R', 'U', 'N', 'N', 1.0, vc_gpu.CholΣ, Z_d, similar(Z_d))
    Z_d
end

"""
	VCM_trait_simulation(mu::Matrix{Float64}, vc::Vector{VarianceComponent})
For an evaluated mean matrix and vector of VarianceComponent objects, simulate from VCM.
"""
function VCM_trait_simulation_gpu(Y_d, μ_d, Z_d, vc_gpu) # for an evaluated matrix
	for i in eachindex(vc_gpu)
		simulate_matrix_normal_gpu!(Z_d, vc_gpu[i]) # this step aggregates the variance components by
		Y_d += Z_d # summing the independent matrix normals to Y, rewriting over Z for each variance component
	end
	Y_d += μ_d # add the mean back to shift the matrix normal
end    
    

 function simulate_gpu!(Y_d, vc_gpu)
      μ_d = CuArray{Float64}(trait.μ)
      Z_d = CuArray{Float64}(trait.Z)
      Y_d .+= VCM_trait_simulation_gpu(Y_d, μ_d, Z_d, vc_gpu)
  end

function simulate_gpu(trait::VCMTrait, vc_gpu, n::Integer)
        # pre-allocate output
        Y_n = [zeros(CuArray{Float32}, size(trait.Z)) for _ in 1:n] # replicates
        # do the simulation n times, each thread needs its own Z to collect randome effects
        for k in 1:n
            simulate_gpu!(Y_n[k], vc_gpu)
        end
        return Y_n
    end

    
# @benchmark vc_gpu = @vc_gpu ΣA ⊗ 2(GRM + 0.01I_n) + ΣE ⊗ I_n;
# @benchmark test_gpu = simulate_gpu(trait, vc_gpu, n_sims);