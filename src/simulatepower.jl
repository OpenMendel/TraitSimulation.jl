using OrdinalMultinomialModels, DataFrames
using VarianceComponentModels, Statistics, Distributions

"""
```
vcm_power(n_sim, γs, traitobject, randomseed)
```
This function aims to design a study around the effect of a causal snp on a VCM trait of interest, controlling for other covariates of interest and family structure.
We use the genetic relationship matrix provided by SnpArrays.jl.
n_obs: number of observations
n_sim: number of simulations
γs: vector of effect sizes of the causal snp (last column of design matrix) to be detected
traitobject: Trait object of type VCMTrait
randomseed: The random seed used for the simulations for reproducible results
"""
function vcm_power(
    nobs, nsim::Int, γs::Vector{Float64}, traitobject::VCMTrait, B_original::AbstractVecOrMat, randomseed::Int)

    #power estimate
    pvalue = Array{Float64}(undef, nsim, length(γs))
    β_original = B_original[end, :]
    Random.seed!(randomseed)

    #generate the data
    X_null = traitobject.X[:, 1:(end - 1)]
    causal_snp = traitobject.X[:, end][:, :]
    μ_original = copy(traitobject.mu)
    μ = traitobject.mu
    y = zeros(size(X_null, 1))
    Σ, V = vcobjtuple(traitobject.vc)
    nulldata = VarianceComponentVariate(y, X_null, (2vcmodel.vc[1].V, vcmodel.vc[2].V))
    altdata = VarianceComponentVariate(y, vcmodel.X, (2vcmodel.vc[1].V, vcmodel.vc[2].V))
    for j in eachindex(γs)
        B_original[end, :] .= γs[j]
        μ .= traitobject.X * B_original
        Y = simulate(traitobject, nsim) # simulate the trait
        for i in 1:nsim
            println(" . . . ")
            copyto!(nulldata.Y, Y[i])
            nullmodel = VarianceComponentModel(nulldata)
            nulllogl, nullmodel, = fit_mle!(nullmodel, nulldata; algo=:FS)

            altmodel = VarianceComponentModel(altdata)
            altlogl, altmodel, = fit_mle!(altmodel, altdata; algo=:FS)

            LRT = 2(altlogl - nulllogl)
            pvalue[i, j] = ccdf(Chisq(1), LRT)
        end
    end
    B_original[end, :] = β_original
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
ordinal_multinomial_power(nsim::Int, γs::Vector{Float64}, traitobject::OrderedMultinomialTrait, randomseed::Int)
```
This function aims to design a study around the effect of a causal snp on an Ordinal trait of interest, controlling for sex and age.
The user can explore the potential of their study design with TraitSimulation.jl input types.
n_sim: number of simulations
γs: vector of effect sizes of the causal snp (last column of design matrix) to be detected
traitobject: this function is type dispatched for the OrdinalTrait type in TraitSimulation, and simulates the trait of the fields of the object.
randomseed: The random seed used for the simulations for reproducible results
"""
function ordinal_multinomial_power(
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
            #simulate the trait
            y = simulate(traitobject)#, Logistic = false, threshold = 0.5)
            #compute the power from the ordinal model
            ornull = polr(X_null, y, traitobject.link)
            pvaluepolr[i, j] = polrtest(OrdinalMultinomialScoreTest(ornull, causal_snp))
        end
    end
    traitobject.β[end] = β_original
    return pvaluepolr
end

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
realistic_multinomial_powers(n, n_sim, maf, θ, γs, meanage, varage, pfemale, link, cutoff, randomseed)
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
function realistic_multinomial_powers(
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
