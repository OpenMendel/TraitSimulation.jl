using Documenter, TraitSimulation

makedocs(
  doctest  = false,
  format   = Documenter.HTML(),
  modules  = [TraitSimulation],
  clean    = true,
  sitename = "TraitSimulation.jl",
  authors  = "Sarah Ji",
  pages = [
    "Home"       => "src/index.md",
    "GLM"   => "src/man/GLM.md",
    "LMM" => "src/man/LMM.md",
    "Example"   => "src/man/RareVariantExample.md",
  ]
)

deploydocs(
  repo   = "github.com/OpenMendel/TraitSimulation.jl.git",
  target = "build",
  deps   = nothing,
  make   = nothing
)