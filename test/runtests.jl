using TraitSimulation
using GLM
using LinearAlgebra
using Random
using Test
using DataFrames
using VarianceComponentModels
using SnpArrays
using Suppressor
using DelimitedFiles

@testset "modelparameters" begin
	include("modelparameters.jl")
end

@testset "simulationmodels" begin
	include("simulationmodels.jl")
	include("matrixnormalsimulation.jl")
	include("writetopheno.jl")
end

@testset "simulatepower" begin
	include("vcm_power.jl")
	include("univariate_power.jl")
end