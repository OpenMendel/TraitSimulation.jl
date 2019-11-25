# import TraitSimulation: simulate_glm_trait

# # test for correct error
# @test_throws ErrorException simulate_glm_trait(-1, TraitSimulation.ExponentialResponse())

# # test for correct summary statistics

# #Create GLMtrait object case for distribution d 

# df = DataFrame(x = repeat([0], 10), y = repeat([1], 10))
# case = GLMTrait(-1, df, TraitSimulation.ExponentialResponse(), TraitSimulation.LogLink())

# #simulate exponential trait 1000 times
# simtraits = simulate(case, 1000)

# # compare summary statistics with simulation summary statistics 
# @test norm(mean(Matrix(simtraits)) - 1/exp(-1)) < 1e-0
# @test norm(var(simtraits) - 1/(exp(-1)^2)) < 1e-0
 



# ## lmm 
# X = DataFrame(snp1 = repeat([0], 10), snp2 = repeat([1], 10))
# formulas = ["1 + snp1", "1 + snp2"]
# npeople = size(X)[1]
# ntraits = 2
# A_1 = [1 0; 0 1]
# B_1 = Matrix{Float64}(I, npeople, npeople)
# A_2 = [1.0 0.0; 0.0 1.0]
# B_2 = Matrix{Float64}(I, npeople, npeople)
# variancecomp1 = @vc A_1 ⊗ B_1

# case = LMMTrait(formulas, X, variancecomp1)

# #simulate exponential trait 1000 times
# simtraits = simulate(case, 1000)

# observed = mean(simtraits, dims = 3)
# expected = [repeat([1], 10) repeat([2], 10)]
# @test all(x -> x < 1e-0, observed .- expected)

 
