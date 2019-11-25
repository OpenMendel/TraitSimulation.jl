using Documenter, TraitSimulation

ENV["DOCUMENTER_DEBUG"] = "true"

makedocs(
    format = Documenter.HTML(),
    sitename = "TraitSimulation",
    authors = "Sarah Ji, Kenneth Lange, Janet Sinsheimer, Eric Sobel",
    modules = [TraitSimulation]
)

deploydocs(
    repo   = "github.com/OpenMendel/TraitSimulation.jl.git",
    target = "build"
)
