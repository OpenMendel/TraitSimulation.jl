using Random, SnpArrays
using LinearAlgebra
Random.seed!(1234)
include("benchmarking.jl")
import TraitSimulation: snparray_simulation
n = 10
p = 2
d = 2
m = 2

df = DataFrame(x = repeat([0], n), y = repeat([1], n))
dist = Normal()
link = IdentityLink()

# test for correct mean formula
formulas = ["x + 5y", "2 + log(y)"]

evaluated_output = [repeat([5], n), repeat([2], n)]

for i in eachindex(formulas)
  return(@test mean_formula(formulas[i], df)[1] == evaluated_output[i])
end

beta = [1, 5]
glmtrait = GLMTrait(Matrix(df), beta, dist, link)

for i in eachindex(evaluated_output[1])
  return(@test GLMTrait(Matrix(df), beta, dist, link).μ[i] == evaluated_output[1][i])
end

# traitversion1 = VCMTrait(X, B, Σ, V)
# variancecomponent = @vc Σ[1] ⊗ V[1] + Σ[2] ⊗ V[2]

X, B, Σ, V = generateRandomVCM(n, p, d, m)
@test VCMTrait(X, B, [Σ...], [V...]).vc[1].Σ - (@vc Σ[1] ⊗ V[1] + Σ[2] ⊗ V[2])[1].Σ == zeros(2, 2)
n, p, d, m = 10, 2 , 2, 2
#testing for types
maf  = 0.2
nsnps = p
@test snparray_simulation([maf], nsnps) isa SnpArrays.SnpArray

effectsizes = rand(n)
our_names = ["sarah"; "janet"; "hua"; "eric"; "ken"; "jenny"; "ben"; "chris"; "juhyun"; "xinkai"]
whats_my_mean_formula = TraitSimulation.FixedEffectTerms(effectsizes, our_names)
data_frame_2 = DataFrame(ones(length(our_names), length(our_names)))
rename!(data_frame_2, Symbol.(our_names))

@test unique(mean_formula(whats_my_mean_formula, data_frame_2)[1])[1] == sum(effectsizes)

variance_formula2  = @vc [maf][:,:] ⊗ V[1] + [maf][:,:] ⊗ V[1]
trait2 = VCMTrait([whats_my_mean_formula], data_frame_2, variance_formula2)

sigma_t, v_t = vcobjtuple(variance_formula2)
mean_formula(whats_my_mean_formula, data_frame_2)[1]

X2, β2, Σ2, V2  = generateRandomVCM(n, p, d, m)
G = snparray_simulation([0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5], n)
genovec = zeros(Float64, size(G))
Base.copyto!(genovec, @view(G[:,:]), model=ADDITIVE_MODEL, impute=true)
Σ2 = [Σ2...]
V2 = [V2...]
γ = rand(7, 2)
varcomp = @vc Σ2[1] ⊗ V2[1] + Σ2[2] ⊗  V2[2]
vcmOBJ =  VCMTrait(X2, β2, genovec, γ, varcomp)

@test vcmOBJ.X == X2
#X*β .+ genovec*γ

vcmOBJ2 =  VCMTrait(X2, β2, varcomp)
@test vcmOBJ2.G == nothing
@test vcmOBJ2.mu ≈ X2*β2 rtol = 0.005
