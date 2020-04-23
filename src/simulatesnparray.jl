using StatsBase, SnpArrays

function snparray_simulation(maf::Vector{Float64}, n::Int)
    genotype = genotype_sim.(maf, n)
    snparray = make_snparray(genotype)
    return snparray
end

# this is for generating genotypes given minor allele frequency under HWE

function genotype_sim(maf::Float64, n::Int)
    pAA = (1 - maf)^2 #0
    pAa = 2maf * (1 - maf) #1
    paa = maf^2     #2
    pvec = [pAA, pAa, paa]
    genes = [0, 1, 2]
    g = StatsBase.wsample(genes, pvec, n)
    return g
end

"""
Make a SnpArray
"""
function make_snparray(genotype)
    n = length(genotype[1])
    p = length(genotype)
    x = SnpArray(undef, n, p)
    for j in 1:p, i in 1:n
        j_locus = genotype[j]
        if j_locus[i] == 0
            x[i, j] = 0x00
        elseif j_locus[i] == 1
            x[i, j] = 0x02
        elseif j_locus[i] == 2
            x[i, j] = 0x03
        else
            throw(error("matrix shouldn't have missing values!"))
        end
    end
    return x
end


# Effect Size Simulation Conditional on MAF
function simulate_effect_size(maf)
    Simulated_ES = ones(length(maf))
    # Generating Effect Sizes where the lower the minor allele frequency the larger the effect size
    for i in eachindex(maf)
        Simulated_ES[i] = rand([-1, 1]) .* (0.01 / sqrt.(maf[i] .* (1 - maf[i])))
    end
    return Simulated_ES
end
