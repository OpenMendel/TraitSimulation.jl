using VarianceComponentModels
using LinearAlgebra
#given genome wide significance of 1 X 10 ^ -8 
# heritability = 0.3 
# snp_i accounts for 1% of the variation in heritability
# maf_i = 0.2
#find power for n = n_families 

#simulate trait under specified alternative 
#simulate trait under null once p-value 0.05 = alpha 
#set n = n_families

function n_families()
#filter data to have the first n_families only
  return(out)
end

function NullFormula(X_covar::DataFrame)
	null_cov_names = String.(names(X_covar))
	for i in length(null_cov_names)

end

function NullFormula(X_covar::DataFrame)
null_cov_names = String.(names(X_covar))
p = length(null_cov_names)
term = "";
for i=1:p
	term = term * "(" * null_cov_names[i] * ")"
    if i != p
        term = term * "+"
    end
end
return(term)
end

# snpdata = SnpArray("traitsim28d.bed", 212)
# famfile = readdlm("traitsim28d.fam", ',')
# height = famfile[:, 7]
# sex = map(x -> strip(x) == "F" ? -1.0 : 1.0, famfile[:, 5])
# snpdef28_1 = readdlm("traitsim28d.bim", Any; header = false)
# snpid = map(x -> strip(string(x)), snpdef28_1[:, 1])

# ind_rs10412915 = findall(x -> x == "rs10412915", snpid)[1]
# locus = convert(Vector{Float64}, @view(snpdata[:,ind_rs10412915]))
# X = DataFrame(sex = sex, locus = locus)

# GRM = grm(snpdata, method= :GRM)
# A_1 = [100 80; 80 100]
# B_1 = GRM
# A_2 = [100 80; 80 100]
# B_2 = Matrix{Float64}(I, size(GRM))

# variancecomp = @vc A_1 ⊗ B_1 + A_2 ⊗ B_2
# trait_null = simulate(LMMTrait(["175", "175"], X, variancecomp))
# formulas = ["175 + 10(sex) + 10(locus)", "60 + 10(sex) + 10(locus)"]
# alternative_model = LMMTrait(formulas, df, variancecomp)
# trait_alternative = simulate(LMMTrait(formulas, X, variancecomp))

#  #export #function  vcobjtuple(vcobject::Vector{VarianceComponent})
# Σ_tuple, V_tuple = vcobjtuple(variancecomp)
# vcdata_null = VarianceComponentVariate(Matrix(trait_null), V_tuple)
# vcdata_alt = VarianceComponentVariate(Matrix(trait_alternative), Matrix(X), V_tuple)
# vcmodel_alt.Σ = Σ_tuple
# vcmodel_mle = deepcopy(vcmodel_alt)
# logl_alt, _, _, _, _, _ = fit_mle!(vcmodel_mle, vcdata_alt; algo = :MM)

# 2*(logl_alt - logl_null) > cquantile(Chisq(1), 0.05)

function PowerSampleSizeCalculation(alternative_formulas, snpdata, vc, X_covar, alpha, B)
	# 1. Generate synthetic data, based on an assumed model
	dataframe = DataFrame(vcat(snpdata, X_covar))
	alternative_model = LMMTrait(alternative_formulas, dataframe, vc)
	
	null_model = LMMTrait(NullFormula(X_covar), dataframe, vc) # with all betas = 0 no genetic variants only intercept and covariates
	trait_null = simulate(null_model)


	Σ_tuple, V_tuple = vcobjtuple(vc)
	vcdata_null = VarianceComponentVariate(Matrix(trait_null), V_tuple) #without covariates
	vcmodel_null = VarianceComponentModel(vcdata_null) # initializes with B= 0 and Sigma = I 
	vcmodel_mle_null = deepcopy(vcmodel_null)
	logl_null, _, _, _, _, _ = fit_mle!(vcmodel_mle_null, vcdata_null; algo = :MM);
	
	rejectH0 = zeros(B)
	for i in 1:B
		trait_i = Matrix(simulate(alternative_model))
		#get loglikelihood
	#2. Fit a model to the synthetic data
	vcdata_alt = VarianceComponentVariate(trait_i, Matrix(dataframe), V_tuple)
	vcmodel_alt = VarianceComponentModel(vcdata_alt)
	vcmodel_alt.Σ = Σ_tuple
	
	vcmodel_mle = deepcopy(vcmodel_alt)
	logl_alt, _, _, _, _, _ = fit_mle!(vcmodel_mle, vcdata_alt; algo = :MM)

	# 3. Do the significance test of interest and record the p-value
		#LRT with null_model 
		if(2*(logl_alt - logl_null) > cquantile(Chisq(1), alpha))
			rejectH0[i] = 1
		else rejectH0[i] = 0

	# 4. repeat 1-3.
		end
	end
	#5. The statistical power is the proportion of p-values that are lower than a specified α-level
	power = sum(rejectH0[i])/ B - 1
	return(power)
end

function PowerSampleSizeCalculation(n_families, fixed_effects_alt, VarianceComponent_alt, alpha, B, trait_null)
	# 1. Generate synthetic data, based on an assumed model

	# a tuple of m covariance matrices
	V = ntuple(x -> zeros(n, n), m) 
	Σ = ntuple(x -> zeros(d, d), m) 


	vcdata_null = VarianceComponentVariate(trait_null, VarianceComponent_alt[2]) #without covariates
	vcmodel_null = VarianceComponentModel(vcdata_null) # initializes with B= 0 and Sigma = I 
	vcmodel_mle_null = deepcopy(vcmodel)
	logl_null, _, _, _, _, _ = fit_mle!(vcmodel_mle_null, vcdata_null; algo = :MM);
	for i in 1:B
		trait_i = Matrix(multiple_trait_simulation7(fixed_effects_mu, VarianceComponent_alt))
		#get loglikelihood
	#2. Fit a model to the synthetic data
	vcdata_alt = VarianceComponentVariate(trait_i, fixed_effects_mu, VarianceComponent_alt[2])
	vcmodel_alt = VarianceComponentModel(ones(n_people, n_cov), vcdata_alt)
		
	# 3. Do the significance test of interest and record the p-value
		#LRT with null_model 
		if(2*(logl_alt - logl_null) > Chisq(1, 1-alpha))
			rejectH0_[i] = 1
		else rejectH0_[i] = 0

	# 4. repeat 1-3.
		end
	end
	#5. The statistical power is the proportion of p-values that are lower than a specified α-level
	power = sum(rejectH0_[i])/ B - 1
	return(power)
end

srand(123)
n = 1000   # no. observations
d = 2      # dimension of responses
m = 3    # no. variance components
p = 2      # no. covariates
# n-by-p design matrix
X = randn(n, p)
# p-by-d mean component regression coefficient
B = ones(p, d)  
# a tuple of m covariance matrices
V = ntuple(x -> zeros(n, n), m) 
Σ = ntuple(x -> zeros(d, d), m) 

	for i in 1:m
		Σi = randn(d, d)
  		copyto!(Σ[i], Σi' * Σi)
	end

for i = 1:m-1
		Vi = randn(n, 50)
  		copyto!(V[i], Vi * Vi')
  		copyto!(V[i], V[i]' * V[i])
	end
	copyto!(V[1], Matrix{Float64}(I, n, n))

function hua_simulation(n, d, m, p, mean, V)

	# form overall nd-by-nd covariance matrix Ω
	Ω = zeros(n * d, n * d)

	for i = 1:m
  		Ω += kron(Σ[i], V[i])
	end

	Ωchol = cholesky(Ω)
	# n-by-d responses
	Y = mean + reshape(Ωchol.L * randn(n*d), n, d)
	return(Y)
end

# test = rand(1000, 1000)
# v = test' * test
# fixedmean = rand(1000, 2)
# VC_huatest = @vc Σ[1] ⊗ v + Σ[2] ⊗ v + Σ[3] ⊗ v
@benchmark hua_simulation(n, d, m, p, fixedmean, V)

@benchmark multiple_trait_simulation7(fixedmean, VC_huatest)