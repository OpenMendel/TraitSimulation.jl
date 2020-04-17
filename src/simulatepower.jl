using OrdinalMultinomialModels, DataFrames, Random
using VarianceComponentModels, Statistics, Distributions
"""
```

power_simulation(n_sim, γs, traitobject, randomseed)
```
This function aims to design a study around the effect of a causal snp on a VCM trait of interest, controlling for other covariates of interest and family structure.
We use the genetic relationship matrix provided by SnpArrays.jl.
n_obs: number of observations
n_sim: number of simulations
γs: vector of effect sizes of the causal snp (last column of design matrix) to be detected
traitobject: Trait object of type VCMTrait
"""
function power_simulation(nsim, γs, traitobject::VCMTrait, B_original; algorithm = :MM)
    pvalue = zeros(nsim, length(γs))
    y_alternative = zeros(size(traitobject.mu))
    μ = traitobject.mu
    tmp_mat = similar(y_alternative)
    tmp_mat2 = similar(y_alternative)
    
    # fit null model once to store nessary information for alternative model 
    nulldata    = VarianceComponentVariate(y_alternative, traitobject.X[:,(end-1)], (2traitobject.vc[1].V, traitobject.vc[2].V))
    nulldatarot = TwoVarCompVariateRotateS(nulldata)
    nullmodel   = VarianceComponentModel(nulldata)

    altdatarot = TwoVarCompVariateRotate(nulldatarot.Yrot, tmp_mat2, nulldatarot.eigval, nulldatarot.eigvec, nulldatarot.logdetV2)
    
    mul!(tmp_mat2, transpose(nulldatarot.eigvec), traitobject.X)
    copyto!(altdatarot.Xrot, tmp_mat2) # last column ramains zero
    altmodel = VarianceComponentModel(altdatarot)
    
    for j in 1:length(γs)
        for k in 1:size(B_original, 2)
            B_original[end, k] = γs[j]
        end
        mul!(μ, traitobject.X, B_original)
        for i in 1:nsim
            TraitSimulation.simulate!(y_alternative, traitobject) # simulate the trait

            # null
            LinearAlgebra.mul!(tmp_mat, transpose(nulldatarot.eigvec), y_alternative)
            copyto!(nulldatarot.Yrot, tmp_mat)


            # alt
            LinearAlgebra.mul!(tmp_mat, transpose(altdatarot.eigvec), y_alternative)
            copyto!(altdatarot.Yrot, tmp_mat)


          # fit null model for the ith simulated trait  
            logl_null, _, _, _, _, _ = mle_mm!(nullmodel, nulldatarot; verbose = false)


            # initialize mean effects to null model fit
            fill!(altmodel.B, zero(eltype(y_alternative)))
            copyto!(altmodel.B, nullmodel.B)
            copyto!(altmodel.Σ[1], nullmodel.Σ[1])
            copyto!(altmodel.Σ[2], nullmodel.Σ[2]) # ask eric and hua tomorrow about this 


            # fit alternative model for ith simulation 
            if algorithm == :MM
                logl_alt, _, _, _, _, _ = mle_mm!(altmodel, altdatarot; verbose = false)
            elseif algorithm == :FS
                logl_alt, = mle_fs!(altmodel, altdatarot; verbose = false)
            end

        #     # LRT statistics and its pvalue
            lrt = - 2(logl_null - logl_alt)
            pvalue[i, j] = ccdf(Chisq(1), lrt)
        end
    end
    return pvalue
end
    
    

"""
```
power_simulation(n_sim, γs, traitobject, randomseed)
```
This function aims to design a study around the effect of a causal snp on a GLM trait of interest, controlling for other covariates of interest.
n_sim: number of simulations
γs: vector of effect sizes of the causal snp (last column of design matrix) to be detected
traitobject: this function is type dispatched for the GLMTrait type in TraitSimulation, and simulates the trait of the fields of the object.
link: link function from GLM.jl
randomseed: The random seed used for the simulations for reproducible results
"""
function power_simulation(
    nsim::Int, γs::Vector{Float64}, traitobject::GLMTrait, randomseed::Int)
    #power estimate
    pvalue = zeros(nsim, length(γs))
    β_original = traitobject.β[end]
    Random.seed!(randomseed)

    #generate the data
    X_null = traitobject.X[:, 1:(end - 1)]
    causal_snp = traitobject.X[:, end]
    for j in eachindex(γs)
        for i in 1:nsim
            β = traitobject.β
            β[end] = γs[j]

            #simulate the trait
            y = simulate(traitobject)
            mu_null = X_null*β[1:(end - 1)]
            locus = causal_snp*β[end]
            ydata = DataFrame(y = y, mu_null = mu_null, locus = traitobject.X[:, end]*β[end]) #for GLM package needs to be in a dataframe
            glm_modelfit = glm(@formula(y ~ mu_null + locus), ydata, traitobject.dist(), traitobject.link)
            pvalue[i, j] = coeftable(glm_modelfit).cols[4][end]
        end
    end
    traitobject.β[end] = β_original
    return pvalue
end

"""
```
ordinal_power_simulation(nsim::Int, γs::Vector{Float64}, traitobject::OrderedMultinomialTrait, randomseed::Int)
```
This function aims to design a study around the effect of a causal snp on an Ordinal trait of interest, controlling for sex and age.
The user can explore the potential of their study design with TraitSimulation.jl input types.
n_sim: number of simulations
γs: vector of effect sizes of the causal snp (last column of design matrix) to be detected
traitobject: this function is type dispatched for the OrdinalTrait type in TraitSimulation, and simulates the trait of the fields of the object.
randomseed: The random seed used for the simulations for reproducible results
"""
function ordinal_power_simulation(
    nsim::Int, γs::Vector{Float64}, traitobject::OrderedMultinomialTrait, randomseed::Int)
    #power estimate
    pvaluepolr = Array{Float64}(undef, nsim, length(γs))
    β_original = traitobject.β[end]
    Random.seed!(randomseed)

    #generate the data
    X_null = traitobject.X[:, 1:(end - 1)]
    causal_snp = traitobject.X[:, end][:, :]
    for j in eachindex(γs)
        for i in 1:nsim
            β = traitobject.β
            β[end] = γs[j]
            y = simulate(traitobject) # simulate the trait
            #compute the power from the ordinal model
            ornull = polr(X_null, y, traitobject.link)
            pvaluepolr[i, j] = polrtest(OrdinalMultinomialScoreTest(ornull, causal_snp))
        end
    end
    traitobject.β[end] = β_original
    return pvaluepolr
end



# function single_ordinal_power(es, traitobject, randomseed, nsim)
#     #generate the data
#     Random.seed!(randomseed)
#     pval = Array{Float64}(undef, nsim)
#     X_null = traitobject.X[:, 1:(end - 1)]
#     causal_snp = traitobject.X[:, end][:, :]
#     for i in 1:nsim
#             β = traitobject.β
#             β[end] = es
#             y = simulate(traitobject) # simulate the trait
#             #compute the power from the ordinal model
#             ornull = polr(X_null, y, traitobject.link)
#             pval[i] = polrtest(OrdinalMultinomialScoreTest(ornull, causal_snp))
#     end
#     pval
# end


# function ordinal_power(gamma::Vector{Float64}, nsim::Int64, alpha::Float64, randomseed::Int64, ordinalmodel::OrderedMultinomialTrait; power = 0.8)
#     # get pvalues from testing the significance of causal snp nsim times
#     pvalues = [single_ordinal_power(ES, ordinalmodel, randomseed, nsim) for ES in gamma]
    
#     # for each row, it represents that effect size, we find the power
#     power_vector = [Statistics.mean(pvalues[i] .< alpha) for i in 1:length(pvalues)]
    
#     #now we want to find where they intersect the desired power value
#     v1 = DataF1(exp.(gamma), power_vector)
#     ze = zeros(length(gamma))
#     vcat(fill!(ze, power)...)
#     v2 = DataF1(exp.(gamma), ze)

#     xings = ycross(v1, v2)
#     min_det_ES = xings.x
#     return pvalues, v1, min_det_ES
# end


"""
```
power(P, α)
```
This function computes the simulated power of the simulation p-values calculated stored in a matrix P, using the ordinal_multinomial_power() function
at the user specified level of significance α.  Specifically, this case is concerned with pvaluelinear, pvaluelogistic, pvaluepolr as input for p1, p2, p3.
"""
function power(P, alpha)
    power_ES = zeros(size(P, 2))
    for i in eachindex(power_ES)
        power_ES[i] = Statistics.mean(P[:, i] .< alpha)
    end
    return power_ES
end

"""
```
ordinal_power_simulation(n, n_sim, maf, θ, γs, meanage, varage, pfemale, link, cutoff, randomseed)
```
This function aims to compare the power under the linear regression, logistic regression and ordered multinomial models.
n: sample size
n_sim: number of simulations
meanage, varage: mean and variance of the age of the study sample of interest
pfemale: proportion of females desired in the study population
link: link function from GLM.jl
cutoff: Which cuttoff to use for Multinomial Traits that may later be transformed to binary 0, 1 Traits (ex. disease vs healthy). If Trait > cutoff -> Trait == 1 else Trait == 0
randomseed: The random seed used for the simulations for reproducible results
"""
function ordinal_power_simulation(
    n::Int, nsim::Int,
    maf::Float64, θ::Vector{Float64}, γs::Float64,
    meanage::Real, varage::Real, pfemale::Real, link, cutoff::Real, randomseed::Int)
    #
    #power estimates
    pvaluelogistic = Array{Float64}(undef, nsim)
    pvaluepolr = Array{Float64}(undef, nsim)
    pvaluelinear = Array{Float64}(undef, nsim)
    # now simulate the genotype with maf p
    d1 = Bernoulli(pfemale)
    d2 = Normal(meanage, varage)

    Random.seed!(randomseed)
    #generate the data
    sex = rand(d1, n)
    age = zscore(rand(d2, n)) #standardize so that it does not produce all 4's for y
    β = [1.0, 1.0, 2.0, γs]
    intercept = ones(n)
    g = genotype_sim(maf, n)
    X = [age sex g]

    for i in 1:nsim
        # simulate the trait
        Ordinal_Model = OrderedMultinomialTrait(X, β, θ, link)
        y = simulate(Ordinal_Model)
        ylogit = Int64.(y .> cutoff) #makes J/2 the default cutoff for case/control
        ydata = DataFrame(y = y, ylogit = ylogit, age = age, sex = sex, g = g) #for GLM package needs to be in a dataframe

        #now compute the power from the different methods
        #logistic
        logit = glm(@formula(ylogit ~ age + sex + g), ydata, Binomial(), LogitLink())
        pvaluelogistic[i] = coeftable(logit).cols[4][end]
        #linear regression
        ols = lm(@formula(y ~ age + sex + g), ydata)
        pvaluelinear[i] = coeftable(ols).cols[4][end]
        #or
        ornull = polr(@formula(y ~ 0 + age + sex), ydata, link)
        pvaluepolr[i] = polrtest(OrdinalMultinomialScoreTest(ornull.model, g[:,:]))
    end
    return pvaluelinear, pvaluelogistic, pvaluepolr
end

"""
```
power_multinomial_models(p1, p2, p3, α)
```
This function computes the simulated power of the simulation p-values calculated using the realistic_multinomial_powers() function,
at the user specified level of significance α.  Specifically, this case is concerned with pvaluelinear, pvaluelogistic, pvaluepolr as input for p1, p2, p3.
"""
function power_multinomial_models(p1, p2, p3, alpha)
    powerlinear = mean(p1 .< alpha)
    powerlogistic = mean(p2 .< alpha)
    powerpolr = mean(p3 .< alpha)
    return(powerlinear, powerlogistic, powerpolr)
end
