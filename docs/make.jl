using Documenter, TraitSimulation

makedocs(
    format = Documenter.HTML(),
    sitename = "TraitSimulation.jl",
    authors = "Sarah Ji",
    clean = true,
    debug = true,
    pages = [
        "index.md"
    ]
)

deploydocs(
    repo   = "github.com/sarah-ji/TraitSimulation.jl.git",
    target = "build",
    deps   = nothing,
    make   = nothing
)