using VarianceComponentModels, Distributions

# set the effect size for covariate being studied
function set_effect_size!(trait, γ)
    trait.β[end, :] .= γ
    mul!(trait.μ, trait.X, trait.β)
    return nothing
end

# rotate the data and pre-compute the eigen decomposition
function null_and_alternative_vcm_and_rotate(Y::AbstractVecOrMat, X::AbstractVecOrMat, V1::AbstractMatrix, V2::AbstractMatrix)
    tmp_mat2 = zeros(size(X))
    vcvar = VarianceComponentVariate(Y, X[:,(end - 1)], (V1, V2))
    vcrot_null = TwoVarCompVariateRotate(vcvar)
    eigen_vecs = transpose(vcrot_null.eigvec)
    vcm_null   = VarianceComponentModel(vcrot_null)
    vcrot_alt = TwoVarCompVariateRotate(vcrot_null.Yrot, tmp_mat2, vcrot_null.eigval, vcrot_null.eigvec, vcrot_null.logdetV2)
    mul!(tmp_mat2, eigen_vecs, X)
    copyto!(vcrot_alt.Xrot, tmp_mat2)
    vcm_alt = VarianceComponentModel(vcrot_alt)
    return vcm_null, vcrot_null, vcm_alt, vcrot_alt
end

# rotate the data
function null_and_alternative_vcm_and_rotate(traitobject::VCMTrait)
    Y = zeros(size(trait.μ))
    X = trait.X
    V1 = trait.vc[1].V
    V2 = trait.vc[2].V
    null_and_alternative_vcm_and_rotate(Y, X, V1, V2)
end

# should modify paramter estimates in vcm_alt
function setup_warm_start!(vcm_alt::VarianceComponentModels.VarianceComponentModel, vcm_null::VarianceComponentModels.VarianceComponentModel)
    copyto!(vcm_alt.B, vcm_null.B)
    copyto!(vcm_alt.Σ[1], vcm_null.Σ[1])
    copyto!(vcm_alt.Σ[2], vcm_null.Σ[2])
    return nothing
end

# should rotate Y data
function rotate_Y_data!(vcrot_null::TwoVarCompVariateRotate, vcrot_alt::TwoVarCompVariateRotate, Y, tmp_mat_Y)
    mul!(tmp_mat_Y, transpose(vcrot_null.eigvec), Y)
    copyto!(vcrot_null.Yrot, tmp_mat_Y)
    copyto!(vcrot_alt.Yrot, tmp_mat_Y)
    vcrot_null, vcrot_alt
end

# function with rotating outside
function power_simulation(trait, γs, nsim, vcm_null, vcrot_null, vcm_alt, vcrot_alt)
    # save original values for trait fields
    β_copy = copy(trait.β)
    μ_copy = copy(trait.μ)
    tmp_mat_Y = zeros(size(μ_copy))
    Y = zeros(size(trait.μ))
    # allocate output --- p-values
    pvalue = zeros(nsim, length(γs))
    for j in eachindex(γs)
        # set the effect size in the trait model
        TraitSimulation.set_effect_size!(trait, γs[j])
        for i in 1:nsim
            # simulate the trait
            simulate!(Y, trait)
            # Y changed, so update rotations for Y
            TraitSimulation.rotate_Y_data!(vcrot_null, vcrot_alt, Y, tmp_mat_Y)
            # fit parameters under null hypothesis model: γ = 0
            fit_null = mle_mm!(vcm_null, vcrot_null; verbose = false)
            logl_null = fit_null[1]
            # use information from null model to start MLE for alternative model
            TraitSimulation.setup_warm_start!(vcm_alt, vcm_null)
            # fit parameters under alternative hypothesis model: γ ≠ 0
            fit_alt = mle_mm!(vcm_alt, vcrot_alt; verbose = false)
            logl_alt = fit_alt[1]
            # compute LRT test statistic; χ² with 1 dof
            lrt = 2 * (logl_alt - logl_null)
            # record p-value for simulation i, effect size j
            pvalue[i, j] = ccdf(Chisq(1), lrt)
        end
    end
    # reset mutated fields for trait object
    copyto!(trait.μ, μ_copy)
    copyto!(trait.β, β_copy)
    return pvalue
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
    mu_null = zeros(size(traitobject.μ))
    mul!(mu_null, X_null, traitobject.β[1:(end - 1)])
    causal_snp = traitobject.G
    y = zeros(size(traitobject.μ))
    for j in eachindex(γs)
        for i in 1:nsim
            β = traitobject.β
            β[end] = γs[j]
            #simulate the trait
            simulate!(y, traitobject)
            ydata = DataFrame(y = y, mu_null = mu_null, locus = causal_snp) #for GLM package needs to be in a dataframe
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
function power_simulation(
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
