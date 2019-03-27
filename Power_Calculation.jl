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


function PowerSampleSizeCalculation(n_families, alternative_formulas, df, vc, alpha, B)
	# 1. Generate synthetic data, based on an assumed model
	alternative_model = LMMTrait(alternative_formulas, df, vc)
	null_model = LMMTrait()
	for i in 1:B
		trait_i = simulate(alternative_model)
		#get loglikelihood
	#2. Fit a model to the synthetic data

		# #LRT with null_model 
		# if(2LRT > Chisq(1, 1-alpha))
		# 	reject[i] = 1
		# else reject[i] = 0

	# 3. Do the significance test of interest and record the p-value


	# 4. repeat 1-3.
		end
	end
	#5. The statistical power is the proportion of p-values that are lower than a specified α-level
	power = sum(rejectH0_[i])/ B - 1
	return(power)
end

#compute estimates of beta hat , variance component, loglikelihood
