using Documenter, TraitSimulation

ENV["DOCUMENTER_DEBUG"] = "true"

makedocs(
    format = Documenter.HTML(),
    sitename = "TraitSimulation.jl",
    authors = "Sarah Ji, Hua Zhou, Kenneth Lange, Janet Sinsheimer, Eric Sobel",
    clean = true,
    debug = true,
    pages = [
    	"index.md"
    ]
)

deploydocs(
    repo   = "github.com/OpenMendel/TraitSimulation.jl.git",
    target = "build"
)

