using Documenter, TraitSimulation

makedocs(
  doctest  = false,
  format   = Documenter.HTML(),
  modules  = [TraitSimulation],
  clean    = true,
  sitename = "TraitSimulation.jl",
  authors  = "Sarah Ji",
  pages = [
    "Home"       => "index.md",
    "GLM"   => "man/GLM.md",
    "LMM" => "man/LMM.md",
    "Example"   => "man/RareVariantExample.md",
  ]
)

deploydocs(
  repo   = "github.com/OpenMendel/TraitSimulation.jl.git",
  target = "build",
  deps   = nothing,
  make   = nothing
)