using SnpArrays, Distributions, Test, Statistics, VarianceComponentModels, Distributions, Random
using DataFrames, LinearAlgebra, StatsBase, TraitSimulation, Distributions
Random.seed!(1234)

# should set the effect size for covariate being studied
function set_effect_size!(trait, γ)
    # set the effect size
    trait.β[end, :] .= γ
    mul!(trait.μ, trait.X, trait.β)
    # make sure μ reflects the correct mean so simulate works correctly
    return nothing
end

function allocate_vcm_and_vcrot(Y, X, V1, V2)
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

# should modify paramter estimates in vcm_alt
function setup_warm_start!(vcm_alt::VarianceComponentModels.VarianceComponentModel, vcm_null::VarianceComponentModels.VarianceComponentModel)
    copyto!(vcm_alt.B, vcm_null.B)
    copyto!(vcm_alt.Σ[1], vcm_null.Σ[1])
    copyto!(vcm_alt.Σ[2], vcm_null.Σ[2])
    return nothing
end

# should rotate Y data
function rotate_Y_data!(vcrot_null::TwoVarCompVariateRotate, vcrot_alt::TwoVarCompVariateRotate, Y)
    tmp_mat_Y = zeros(size(Y))
    mul!(tmp_mat_Y, transpose(vcrot_null.eigvec), Y)
    copyto!(vcrot_null.Yrot, tmp_mat_Y)
    copyto!(vcrot_alt.Yrot, tmp_mat_Y)
    vcrot_null, vcrot_alt
end

function power_simulation(trait::VCMTrait, γs, nsim, vcm_null, vcrot_null, vcm_alt, vcrot_alt)
    # save original values for trait fields
    β_copy = copy(trait.β)
    μ_copy = copy(trait.μ)

    # allocate output --- p-values
    pvalue = zeros(nsim, length(γs))
    Y = zeros(size(trait.μ))
    for j in eachindex(γs)
        # set the effect size in the trait model
        set_effect_size!(trait, γs[j])
        for i in 1:nsim
            # simulate the trait
            simulate!(Y, trait)

            # Y changed, so update rotations for Y
            rotate_Y_data!(vcrot_null, vcrot_alt, Y)

            # fit parameters under null hypothesis model: γ = 0
            fit_null = mle_mm!(vcm_null, vcrot_null; verbose = false)
            logl_null = fit_null[1]

            # use information from null model to start MLE for alternative model
            setup_warm_start!(vcm_alt, vcm_null)

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

# demo genetic data can be found in Snparrays package directory
filename = "EUR_subset"
EUR = SnpArray(SnpArrays.datadir(filename * ".bed"));
GRM = grm(EUR, minmaf = 0.05);
EUR_data = SnpData(SnpArrays.datadir(filename));
bimfile = EUR_data.snp_info # store the snp_info with the snp names
snpid  = bimfile[:, :snpid] # store the snp names in the snpid vector
causal_snp_index = findfirst(x -> x ==  "rs62057050" , snpid) # find the index of the snp of interest by snpid

locus = convert(Vector{Float64}, @view EUR[:, causal_snp_index])
famfile = EUR_data.person_info
sex = map(x -> strip(x) == "F" ? 0.0 : 1.0, famfile[!, :sex])
X = [sex locus]
β_full =
 [0.52
0.05]

I_n = Matrix(I, size(GRM))
vc = @vc [0.01][:,:] ⊗ (GRM + I_n) + [0.9][:, :] ⊗ I_n
trait = VCMTrait(X, β_full[:, :], vc)
X = trait.X
V1 = trait.vc[1].V
V2 = trait.vc[2].V

nsim = 2
nsim = 10
γs = collect(0.0:0.05:0.4)
Y = zeros(size(trait.μ))
X = trait.X
V1 = trait.vc[1].V
V2 = trait.vc[2].V
@test eltype(Y) == Float64

# allocate variance component objects for null and alternative hypothesis
vcm_null, vcrot_null, vcm_alt, vcrot_alt = @time allocate_vcm_and_vcrot(Y, X, V1, V2)
pvalue = power_simulation(trait, γs, nsim, vcm_null, vcrot_null, vcm_alt, vcrot_alt)

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

alpha = 0.0000005
powers = power(pvalue, alpha)
@test powers[1] == 0
@test powers[1] < powers[end]

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
