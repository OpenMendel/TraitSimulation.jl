using Documenter, TraitSimulation

ENV["DOCUMENTER_DEBUG"] = "true"

makedocs(
    format = Documenter.HTML(),
    sitename = "TraitSimulation.jl",
    authors = "Sarah Ji, Hua Zhou, Kenneth Lange, Janet Sinsheimer, Eric Sobel",
    clean = true,
    debug = true,
    pages = Any[
        "Home" => "index.md",
	"Testing Methods" => "examples/testing_MendelIHT_glm.md",
        "Real Data Examples" => Any[
            "examples/ukbiobank_vcm_power.md",
	    "examples/ukbiobank_ordered_multinomial_power.md"],
	"Exploring GPU Potential" => "examples/GPU.md",
     ]
)

deploydocs(
    repo   = "github.com/OpenMendel/TraitSimulation.jl.git",
    target = "build"
)

