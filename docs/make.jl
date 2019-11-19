using Documenter, TraitSimulation

makedocs(
    format = Documenter.HTML(),
    sitename = "TraitSimulation.jl",
    authors = "Sarah Ji, Kenneth Lange, Janet Sinsheimer, Eric Sobel",
    clean = true,
    debug = true,
    pages = [
        "index.md"
    ]
)

deploydocs(
    repo   = "github.com/OpenMendel/TraitSimulation.jl.git",
    target = "build",
    deps   = nothing,
    make   = nothing
)