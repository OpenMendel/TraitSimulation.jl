"""
writepheno
writepheno will simulate the traits under the given model for GLM and OrderedMultinomial traits (univariate) and write the simulation results to tab separated PLINK formatted phenotype file titled "pheno.txt". 
The first two columns of the phenotype file will be the Family ID and the Individual ID the working directory `file_path`, and will find the family information stored in the ".fam" file in this data directory.
"""
function writepheno(plink_file::AbstractString, file_path::AbstractString, traitobject::Union{GLMTrait, OrderedMultinomialTrait}, nsim::Int; famname::AbstractString=plink_file*".fam")
    # simulate phenotypes
    pheno = DataFrame(simulate(traitobject, nsim), :auto)
    rename!(pheno, [Symbol("Trait$i") for i in 1:nsim])

    # load person info
    dlmdata = readdlm(file_path*"/"*famname, AbstractString)
    famfile = DataFrame(Tables.table(dlmdata, header = Symbol.(:x, axes(dlmdata,2))))
    personinfo = famfile[:, 1:2]
    outfile = hcat(personinfo, pheno)
    open("pheno.txt"; write=true) do f
        write(f, "#FID, IID, $nsim simulations replications\n")
        writedlm(f, Matrix(outfile))
    end
end

"""
writepheno
writepheno will simulate the traits under the given model for Variance Component Model traits and GLMM traits (multivariate) and write the simulation results to tab separated PLINK formatted phenotype file titled "pheno.txt". 
The first two columns of the phenotype file will be the Family ID and the Individual ID the working directory `file_path`, and will find the family information stored in the ".fam" file in this data directory.
"""
function writepheno(plink_file::AbstractString, file_path::AbstractString, traitobject::Union{VCMTrait, GLMMTrait}, nsim::Int; famname::AbstractString=plink_file*".fam")
    # simulate phenotypes
    n_traits = ntraits(traitobject)
    phenotypes = simulate(traitobject, nsim)
    pheno = [phenotypes[i][:, j] for j in 1:n_traits, i in 1:nsim]
    pheno = DataFrame(hcat(pheno...), :auto)
    rename!(pheno[:,1:n_traits:n_traits*nsim], [Symbol("Simulation$i") for i in 1:nsim])

    # load person info
    dlmdata = readdlm(file_path*"/"*famname, AbstractString)
    famfile = DataFrame(Tables.table(dlmdata, header = Symbol.(:x, axes(dlmdata,2))))
    personinfo = famfile[:, 1:2]
    rename!(personinfo, [Symbol("FID"), Symbol("IID")])

    outfile = hcat(personinfo, pheno)

    open("pheno.txt"; write=true) do f
        write(f, "#FID, IID, $nsim simulations replications for $n_traits traits\n")
        writedlm(f, Matrix(outfile))
    end
end