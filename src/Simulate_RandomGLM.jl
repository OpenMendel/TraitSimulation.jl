# input x:snparrays(not converted to float), snpid:Vector(string), indexvector_k 
function Generate_Random_Model_Chisq(filepath::String, k::Int64; effectsizes = [], min_success_rate_per_row = 0.98, min_success_rate_per_col = 0.98, min_maf = 0.01, min_hwe_pval = 0.0, maxiters = 5)
snpdata = SnpArray(filepath * ".bed")
rowmask, colmask =  SnpArrays.filter(snpdata,
										min_success_rate_per_row = min_success_rate_per_row,
										min_success_rate_per_col = min_success_rate_per_col,
										min_maf = min_maf, min_hwe_pval = min_hwe_pval,
										maxiters = maxiters)

x = SnpArrays.filter(filepath, rowmask, colmask)

SNP_data = SnpData(filepath)
snpid = SNP_data.snp_info[:, :snpid]
k_indices = rand(1:length(snpid), k)
snpid_k = snpid[k_indices]

genotype_converted = DataFrame(convert(Matrix{Float64}, x[:, k_indices]));
names!(genotype_converted, Symbol.(snpid_k))

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


meanformula_k = FixedEffectTerms(Simulated_ES, snpid_k)

return(meanformula_k, genotype_converted)
end


# meanformula_k, genotype_df = Generate_Random_Model_Chisq(filepath, k)
# model_k = GLMTrait(meanformula_k, genotype_df, NormalResponse(1), IdentityLink())
# simulated_trait = simulate(model_k)
