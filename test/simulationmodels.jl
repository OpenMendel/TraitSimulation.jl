using Random, SnpArrays, DataFrames, GLM
using LinearAlgebra
using BenchmarkTools
beta = [1.0, 5.0]
glmtraitobject = GLMTrait(Matrix(df), beta, dist, link)
y_normal  = simulate(glmtraitobject)
@test size(y_normal) == size(Matrix(df)*beta)

number_independent_simulations  = 5
@test size(simulate(glmtraitobject, number_independent_simulations)) == (n, number_independent_simulations)

@test glmtraitobject isa TraitSimulation.AbstractTraitModel

@test GLMTrait(Matrix(df), beta, dist, link).η == evaluated_output[1]

# make sure that ordered multinomial models works
link = LogitLink()
θ = [1.0, 1.2, 1.4]
Ordinal_Model_Test = OrderedMultinomialTrait(X, beta, θ, link)

@test Ordinal_Model_Test isa TraitSimulation.AbstractTraitModel
@test eltype(simulate(Ordinal_Model_Test)) == Int64
@test size(simulate(Ordinal_Model_Test, number_independent_simulations))  == (n, number_independent_simulations)

# make sure GLM simulation works
dist = Poisson()
link = IdentityLink()
beta = [21.0, 5.0]
glmtraitobject2 = GLMTrait(abs.(X2), beta, dist, link)

# check if clamper worked
@test sum(glmtraitobject2.η .> 20.0) == 0

@test  eltype(simulate(glmtraitobject2)) == Int64

dist = Bernoulli()
link = LogitLink()
glmtraitobject3 = GLMTrait(X2, beta, dist, link)

@test eltype(simulate(glmtraitobject3)) == Bool

@test size(simulate(glmtraitobject3, number_independent_simulations))  == (n, number_independent_simulations)

dist = NegativeBinomial()
link = LogLink()
glmtraitobject4 = GLMTrait(X2, beta, dist, link)

@test eltype(simulate(glmtraitobject4)) == Int64

@test size(simulate(glmtraitobject4, number_independent_simulations))  == (n, number_independent_simulations)

dist = Gamma()
link = LogLink()
glmtraitobject5 = GLMTrait(X2, beta, dist, link)

@test eltype(simulate(glmtraitobject5)) == Float64

@test size(simulate(glmtraitobject5, number_independent_simulations))  == (n, number_independent_simulations)

# make sure GLMM works
glmm_test = GLMMTrait(X2, β2, varcomp, dist, link)
@test glmm_test isa TraitSimulation.AbstractTraitModel

y_glmm = simulate(glmm_test)
@test size(y_glmm) == (size(X2,  1),  size(β2, 2))

@test length(simulate(glmm_test, number_independent_simulations)) == number_independent_simulations
