using TraitSimulation
using GLM
using LinearAlgebra
using Random
using Test
using DataFrames
using VarianceComponentModels
using SnpArrays
using Suppressor

@testset "modelparameters" begin
	include("modelparameters.jl")
end

@testset "matrixnormalsimulation" begin
	include("matrixnormalsimulation.jl")
end

@testset "simulationmodels" begin
	include("simulationmodels.jl")
end

@testset "simulatepower" begin
	include("vcm_power.jl")
	include("univariate_power.jl")
end
