using SnpArrays

"""
Generate_Random_Model_Chisq(filepath, k; effectsizes, min_success_rate_per_row, min_maf, min_hwe_pval, maxiters)

This function allows user to specify the filepath, the number of snps for simulation of the phenotype.

For demonstration purposes, we simulate effect sizes from the Chi-squared(df = 1) distribution,
where we use the minor allele frequency (maf) as x and find f(x),
where f is the pdf for the Chi-squared (df = 1) density,
so that the rarest SNP's have the biggest effect sizes.
The effect sizes are rounded to the second digit, throughout this example.
Notice there is a random +1 or -1, so that there are effects that both increase and decrease the simulated trait value.
"""
function Generate_Random_Model_Chisq(filepath::String, k::Int64; effectsizes = [], min_success_rate_per_row = 0.98, min_success_rate_per_col = 0.98, min_maf = 0.01, min_hwe_pval = 0.0, maxiters = 5)
	
	#read in the data using SnpArrays
	snpdata = SnpArray(filepath * ".bed")

	# find the row and column indices to filter according to the specified arguments
	rowmask, colmask =  SnpArrays.filter(snpdata,
											min_success_rate_per_row = min_success_rate_per_row,
											min_success_rate_per_col = min_success_rate_per_col,
											min_maf = min_maf, min_hwe_pval = min_hwe_pval,
											maxiters = maxiters)

	# use SnpArrays.jl's filtering function to save these as plink outputs
	x = SnpArrays.filter(filepath, rowmask, colmask)

	# use the SnpData function in SnpArrays to store the person and snp info
	SNP_data = SnpData(filepath)
	snpid = SNP_data.snp_info[!, :snpid]
	snp_position = SNP_data.snp_info[!, :position]
	
	#this is the total number of snps in our filtered sample
	n_filt_snps = sum(colmask)
	# randomly choose k of the indices (snps)
	k_indices = rand(1:n_filt_snps, k)

	#find the snp id and position of the k chosen snps
	snpid_k = snpid[colmask][k_indices]
	snp_posk = snp_position[colmask][k_indices]

	#convert and store as a dataframe with column names the snpid names of the k chosen snps
	genotype_converted = DataFrame(convert(Matrix{Float64}, @view(x[:, k_indices])));
	names!(genotype_converted, Symbol.(snpid_k))

	#if the user doesn't specify effect sizes we generate them from a chisquared distribution and the maf
	Simulated_ES = ones(k)
	if isempty(effectsizes)
		# Generating Effect Sizes from theoretical Chisquared(df = 1) density, where the lower the minor allele frequency the larger the effect size
		maf_k_snps = maf(x)[k_indices]
		for i in 1:k
			Simulated_ES[i] = rand([-1, 1]) .* (0.1 / sqrt.(maf_k_snps[i] .* (1 - maf_k_snps[i])))
		end
	else
		copyto!(Simulated_ES, effectsizes)
	end

	#use the fixed effect terms to get the mean formula
	meanformula_k = FixedEffectTerms(Simulated_ES, snpid_k)

	return(meanformula_k, snpid_k, snp_posk, Simulated_ES, genotype_converted)
end

# simulation
# meanformula_k, genotype_df = Generate_Random_Model_Chisq(filepath, k)
# model_k = GLMTrait(meanformula_k, genotype_df, NormalResponse(1), IdentityLink())
# simulated_trait = simulate(model_k)
