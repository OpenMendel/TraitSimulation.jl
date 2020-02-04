using OrdinalMultinomialModels

"""
```
realistic_multinomial_powers(n, n_sim, maf, θ, γs, meanage, varage, pfemale, link, cutoff, randomseed)
```
This function aims to design a study around the effect of a causal snp on an Ordinal trait of interest, controlling for sex and age.
The user can explore the potential of their study design with different specified parameters.
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
    #say a is the minor allele, then we will generate the genotype vector according to hardy weinberg equilibrium
    # now simulate the genotype with maf p
    d1 = Bernoulli(pfemale)
    d2 = Normal(meanage, varage)

    Random.seed!(randomseed)
    for i in 1:nsim
        #generate the data
        sex = rand(d1, n)
        age = zscore(rand(d2, n)) #standardize so that it does not produce all 4's for y
        β = [1.0, 2.0, γs]
        g = genotype_sim(maf, n)
        X = [age sex g]

        # simulate the trait
        Ordinal_Model = OrdinalTrait(X, β, θ, link)
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
