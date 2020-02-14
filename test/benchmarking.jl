# using Distributions, BenchmarkTools
#
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

### First using the TraitSimulation.jl package


function MN_J(X, B, Σ, V; n_reps = 1)
    n, p = size(X*B)
    sim = [zeros(n, p) for i in 1:n_reps]
    for i in 1:n_reps
        sim[i] = rand(MatrixNormal(X*B, V, Σ))
    end
    return(sim)
end

### Now using the Distributions.jl package
function MN_Jm(X, B, Σ, V; n_reps = 1)
    n, p = size(X*B)
    m = length(V)
    sim = [zeros(n, p) for i in 1:n_reps]
    for i in 1:n_reps
        for j in 1:m
            dist = MatrixNormal(X*B, V[j], Σ[j])
            sim[i] += rand(dist)
        end
    end
    return(sim)
end

function CompareWithJulia(X, B, Σ, V; n_reps = 1)
	d = size(B, 2)
	vecVC = [VarianceComponent(Σ[i], V[i]) for i in eachindex(V)]
	LMMtraitobjm = VCMTrait(X*B, vecVC)
	m1trait = simulate(LMMtraitobjm, n_reps)

	m1julia = MN_Jm(X, B, Σ, V; n_reps = n_reps)[:]

	m1traitall = [collect(m1trait[:, i, :])[:] for i in 1:d]

	m1traitlong = hcat(m1traitall...)
	m1julialong = vcat(m1julia...)

	dfwrite = DataFrame(hcat(m1traitlong, m1julialong))
	return(dfwrite)
end
